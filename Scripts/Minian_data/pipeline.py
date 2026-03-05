"""
Minian pipeline – standalone script

--------------------------------------------------------------------------------------------------
!!!!! IMPORTANT !!!!!!
This script still needs to be run in the necessary conda environment. Due to complex dependencies. 
--------------------------------------------------------------------------------------------------
Converted from pipeline.ipynb.

Usage
-----
    # Run directly (set dpath below in __main__ block)
    python pipeline.py

    # Or import and call from low_level_loop.py
    from Scripts.Minian_data.pipeline import run_pipeline
    run_pipeline("/scratch/s4750098/session_001")

Requirements
------------
    - minian installed / available on sys.path (minian_path below)
    - dask, holoviews, xarray, numpy installed
    - Scripts convert_to_csv.py, plotting.py, map.py in the same folder as this file
"""

import itertools as itt
import os
import sys
import shutil as _shutil
import time

import holoviews as hv
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
from holoviews.operation.datashader import datashade, regrid
from holoviews.util import Dynamic
from IPython.core.display import display

# ---------------------------------------------------------------------------
# PATH TO MINIAN CODEBASE
# Default: "." (same folder as this file). Override from low_level_loop.py:
#     import pipeline; pipeline.MINIAN_PATH = "/path/to/minian"
# ---------------------------------------------------------------------------
MINIAN_PATH = "."


def run_pipeline(dpath: str) -> None:
    """
    Run the full Minian CNMF pipeline on a single session folder.

    Parameters
    ----------
    dpath : str
        Path to the local folder that contains the raw .avi videos.
        When called from low_level_loop.py this is the /scratch download path.
    """

    pipeline_start = time.time()

    # -----------------------------------------------------------------------
    # Add minian to path and import
    # -----------------------------------------------------------------------
    sys.path.append(MINIAN_PATH)
    from minian.cnmf import (
        compute_AtC,
        compute_trace,
        get_noise_fft,
        smooth_sig,
        unit_merge,
        update_spatial,
        update_temporal,
        update_background,
    )
    from minian.initialization import (
        gmm_refine,
        initA,
        initC,
        intensity_refine,
        ks_refine,
        pnr_refine,
        seeds_init,
        seeds_merge,
    )
    from minian.motion_correction import apply_transform, estimate_motion
    from minian.preprocessing import denoise, remove_background
    from minian.utilities import (
        TaskAnnotation,
        get_optimal_chk,
        load_videos,
        open_minian,
        save_minian,
    )
    from minian.visualization import (
        CNMFViewer,
        VArrayViewer,
        generate_videos,
        visualize_gmm_fit,
        visualize_motion,
        visualize_preprocess,
        visualize_seeds,
        visualize_spatial_update,
        visualize_temporal_update,
        write_video,
    )

    # -----------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------
    dpath = os.path.abspath(dpath)
    minian_ds_path = os.path.join(dpath, "minian")
    intpath = "./minian_intermediate"
    subset = dict(frame=slice(0, None))
    subset_mc = None
    interactive = False  # set True only for interactive parameter exploration, Hábrok does not have built in GUI support so only locally runable with interactive
    output_size = 100
    n_workers = int(os.getenv("MINIAN_NWORKERS", 4))

    param_save_minian = {
        "dpath": minian_ds_path,
        "meta_dict": dict(session=-1, animal=-2),
        "overwrite": True,
    }

    # Pre-processing
    param_load_videos = {
        "pattern": r"^msCam([1-9]|1[0-9]|2[0-8])\.avi$",
        "dtype": np.uint8,
        "downsample": dict(frame=1, height=6, width=6),
        "downsample_strategy": "subset",
    }
    param_denoise = {"method": "median", "ksize": 7}
    param_background_removal = {"method": "tophat", "wnd": 15}

    # Motion correction
    param_estimate_motion = {"dim": "frame"}

    # Initialization
    param_seeds_init = {
        "wnd_size": 1000,
        "method": "rolling",
        "stp_size": 500,
        "max_wnd": 25,
        "diff_thres": 3,
    }
    param_pnr_refine = {"noise_freq": 0.06, "thres": 0.9}
    param_ks_refine = {"sig": 0.05}
    param_seeds_merge = {"thres_dist": 7, "thres_corr": 0.8, "noise_freq": 0.06}
    param_initialize = {"thres_corr": 0.8, "wnd": 15, "noise_freq": 0.1}
    param_init_merge = {"thres_corr": 0.9}

    # CNMF
    param_get_noise = {"noise_range": (0.06, 0.5)}
    param_first_spatial = {
        "dl_wnd": 25,
        "sparse_penal": 0.01,
        "size_thres": (20, None),
    }
    param_first_temporal = {
        "noise_freq": 0.06,
        "sparse_penal": 1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.2,
    }
    param_first_merge = {"thres_corr": 0.9}
    param_second_spatial = {
        "dl_wnd": 15,
        "sparse_penal": 0.001,
        "size_thres": (15, None),
    }
    param_second_temporal = {
        "noise_freq": 0.06,
        "sparse_penal": 1,
        "p": 1,
        "add_lag": 20,
        "jac_thres": 0.4,
    }

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MINIAN_INTERMEDIATE"] = intpath

    # -----------------------------------------------------------------------
    # Module initialisation
    # -----------------------------------------------------------------------
    hv.extension("bokeh", logo=False)

    # -----------------------------------------------------------------------
    # Start Dask cluster
    # -----------------------------------------------------------------------
    print("[pipeline] Starting Dask cluster ...")
    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit="7GB",
        resources={"MEM": 1},
        threads_per_worker=4,
        dashboard_address=":8787",
        processes=True,
    )
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)
    print(f"[pipeline] Dask dashboard: {client.dashboard_link}")

    try:

        # ===================================================================
        # PRE-PROCESSING
        # ===================================================================
        print("[pipeline] Loading videos ...")
        varr = load_videos(dpath, **param_load_videos)
        chk, _ = get_optimal_chk(varr, dtype=float)

        varr = save_minian(
            varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
            intpath,
            overwrite=True,
        )

        # Subset
        varr_ref = varr.sel(subset)

        # Glow removal
        print("[pipeline] Glow removal ...")
        varr_min = varr_ref.min("frame").compute()
        varr_ref = varr_ref - varr_min

        # Denoise
        print("[pipeline] Denoising ...")
        varr_ref = denoise(varr_ref, **param_denoise)

        # Background removal
        print("[pipeline] Background removal ...")
        varr_ref = remove_background(varr_ref, **param_background_removal)

        # Save pre-processed video
        varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)

        # ===================================================================
        # MOTION CORRECTION
        # ===================================================================
        print("[pipeline] Estimating motion ...")
        motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)
        motion = save_minian(
            motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian
        )

        print("[pipeline] Applying motion correction ...")
        Y = apply_transform(varr_ref, motion, fill=0)

        Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
        Y_hw_chk = save_minian(
            Y_fm_chk.rename("Y_hw_chk"),
            intpath,
            overwrite=True,
            chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
        )

        # Make motion-correction comparison video
        vid_arr = xr.concat([varr_ref, Y_fm_chk], "width").chunk({"width": -1})
        write_video(vid_arr, "minian_mc.mp4", dpath)

        # ===================================================================
        # INITIALIZATION
        # ===================================================================
        print("[pipeline] Computing max projection ...")
        max_proj = save_minian(
            Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian
        ).compute()

        print("[pipeline] Generating seeds ...")
        seeds = seeds_init(Y_fm_chk, **param_seeds_init)

        print("[pipeline] PNR refine ...")
        seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)

        print("[pipeline] KS refine ...")
        seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)

        print("[pipeline] Merging seeds ...")
        seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
        seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)

        print("[pipeline] Initialising spatial matrix ...")
        A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
        A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)

        print("[pipeline] Initialising temporal matrix ...")
        C_init = initC(Y_fm_chk, A_init)
        C_init = save_minian(
            C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1}
        )

        print("[pipeline] Merging units (init) ...")
        A, C = unit_merge(A_init, C_init, **param_init_merge)
        A = save_minian(A.rename("A"), intpath, overwrite=True)
        C = save_minian(C.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(
            C.rename("C_chk"),
            intpath,
            overwrite=True,
            chunks={"unit_id": -1, "frame": chk["frame"]},
        )

        print("[pipeline] Initialising background terms ...")
        b, f = update_background(Y_fm_chk, A, C_chk)
        f = save_minian(f.rename("f"), intpath, overwrite=True)
        b = save_minian(b.rename("b"), intpath, overwrite=True)

        # ===================================================================
        # CNMF
        # ===================================================================

        # --- Estimate spatial noise ---
        print("[pipeline] Estimating spatial noise ...")
        sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
        sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

        # -------------------------------------------------------------------
        # First spatial update
        # -------------------------------------------------------------------
        print("[pipeline] First spatial update ...")
        A_new, mask, norm_fac = update_spatial(
            Y_hw_chk, A, C, sn_spatial, **param_first_spatial
        )
        C_new = save_minian(
            (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
        )
        C_chk_new = save_minian(
            (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True
        )
        b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

        A = save_minian(
            A_new.rename("A"),
            intpath,
            overwrite=True,
            chunks={"unit_id": 1, "height": -1, "width": -1},
        )
        b = save_minian(b_new.rename("b"), intpath, overwrite=True)
        f = save_minian(
            f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
        )
        C = save_minian(C_new.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

        # -------------------------------------------------------------------
        # First temporal update
        # -------------------------------------------------------------------
        print("[pipeline] Computing trace (1st temporal) ...")
        YrA = save_minian(
            compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
            intpath,
            overwrite=True,
            chunks={"unit_id": 1, "frame": -1},
        )

        print("[pipeline] First temporal update ...")
        # Use a clean sub-directory for intermediate results so open_minian()
        # only finds valid zarr stores during this step
        _intpath_orig = intpath
        intpath = os.path.join(_intpath_orig, "_tmp_update_temporal")
        if os.path.exists(intpath):
            _shutil.rmtree(intpath)
        os.makedirs(intpath)
        os.environ["MINIAN_INTERMEDIATE"] = intpath

        try:
            C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
                A, C, YrA=YrA, **param_first_temporal
            )
        finally:
            intpath = _intpath_orig
            os.environ["MINIAN_INTERMEDIATE"] = intpath

        C = save_minian(
            C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
        )
        C_chk = save_minian(
            C.rename("C_chk"),
            intpath,
            overwrite=True,
            chunks={"unit_id": -1, "frame": chk["frame"]},
        )
        S = save_minian(
            S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
        )
        b0 = save_minian(
            b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
        )
        c0 = save_minian(
            c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
        )
        A = A.sel(unit_id=C.coords["unit_id"].values)

        # -------------------------------------------------------------------
        # Merge units (between CNMF iterations)
        # -------------------------------------------------------------------
        print("[pipeline] Merging units (first merge) ...")
        A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)

        A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
        C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
        C_chk = save_minian(
            C.rename("C_mrg_chk"),
            intpath,
            overwrite=True,
            chunks={"unit_id": -1, "frame": chk["frame"]},
        )
        sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

        # -------------------------------------------------------------------
        # Second spatial update
        # -------------------------------------------------------------------
        print("[pipeline] Second spatial update ...")
        A_new, mask, norm_fac = update_spatial(
            Y_hw_chk, A, C, sn_spatial, **param_second_spatial
        )
        C_new = save_minian(
            (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
        )
        C_chk_new = save_minian(
            (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True
        )
        b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

        A = save_minian(
            A_new.rename("A"),
            intpath,
            overwrite=True,
            chunks={"unit_id": 1, "height": -1, "width": -1},
        )
        b = save_minian(b_new.rename("b"), intpath, overwrite=True)
        f = save_minian(
            f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
        )
        C = save_minian(C_new.rename("C"), intpath, overwrite=True)
        C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

        # -------------------------------------------------------------------
        # Second temporal update
        # -------------------------------------------------------------------
        print("[pipeline] Computing trace (2nd temporal) ...")
        YrA = save_minian(
            compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"),
            intpath,
            overwrite=True,
            chunks={"unit_id": 1, "frame": -1},
        )

        print("[pipeline] Second temporal update ...")
        C_new, S_new, b0_new, c0_new, g, mask = update_temporal(
            A, C, YrA=YrA, **param_second_temporal
        )

        C = save_minian(
            C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
        )
        C_chk = save_minian(
            C.rename("C_chk"),
            intpath,
            overwrite=True,
            chunks={"unit_id": -1, "frame": chk["frame"]},
        )
        S = save_minian(
            S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
        )
        b0 = save_minian(
            b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
        )
        c0 = save_minian(
            c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True
        )
        A = A.sel(unit_id=C.coords["unit_id"].values)

        # ===================================================================
        # FINAL VISUALISATIONS & SAVE
        # ===================================================================
        print("[pipeline] Generating output video ...")
        generate_videos(varr.sel(subset), Y_fm_chk, A=A, C=C_chk, vpath=dpath)

        print("[pipeline] Saving final results ...")
        A  = save_minian(A.rename("A"),   **param_save_minian)
        C  = save_minian(C.rename("C"),   **param_save_minian)
        S  = save_minian(S.rename("S"),   **param_save_minian)
        c0 = save_minian(c0.rename("c0"), **param_save_minian)
        b0 = save_minian(b0.rename("b0"), **param_save_minian)
        b  = save_minian(b.rename("b"),   **param_save_minian)
        f  = save_minian(f.rename("f"),   **param_save_minian)

        # ===================================================================
        # POST-PROCESSING SCRIPTS (convert_to_csv, plotting, map)
        # ===================================================================
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        for script_name in ("convert_to_csv.py", "plotting.py", "map.py"):
            script_path = os.path.join(_script_dir, script_name)
            if os.path.exists(script_path):
                print(f"[pipeline] Running {script_name} ...")
                exec(open(script_path).read(), {"dpath": dpath})
            else:
                print(f"[pipeline] WARNING: {script_name} not found at {script_path}, skipping.")

    finally:
        # Always close cluster, even if something fails mid-pipeline
        print("[pipeline] Closing Dask cluster ...")
        client.close()
        cluster.close()

    pipeline_end = time.time()
    total_runtime = pipeline_end - pipeline_start
    print(
        f"[pipeline] Done. Total runtime: {total_runtime:.2f}s "
        f"({total_runtime / 60:.2f} min)"
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Change this path to run the pipeline standalone
    run_pipeline("24_20190905_T2")
