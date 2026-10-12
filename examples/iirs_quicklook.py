"""Quick-look figure for one Chandrayaan-2 IIRS product: band images, spectra and striping.

Works on raw DN (``nri``) products; for ``nci`` / ``ndi`` products only the
y-axis meaning changes. Needs matplotlib (``pip install lunar-ir-data-tools[plot]``).

Usage::

    python examples/iirs_quicklook.py <bundle_dir_or_qub> --out <output_dir>
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lunar_ir_tools import IirsProduct  # noqa: E402

QL_WAVELENGTHS = (1005, 1555, 2500)   # nm; band-image panels
LINE_DS = 8                           # along-track downsample for the panels


def quicklook(product: str, out_dir: str) -> str:
    p = IirsProduct(product)
    c = p.cube
    nb, nl, ns = p.label["shape"]
    try:                                   # OAT-based per-line latitude, if the bundle has it
        ll = p.line_latlon()
        orbit = "/".join(map(str, np.unique(p.oat()["orbit"])))
        xlab, x0, x1 = "boresight latitude (deg)", ll[0, 0], ll[-1, 0]
        where = "lat %.2f..%.2f  lon~%.2f  orbit %s" % (ll[0, 0], ll[-1, 0], np.median(ll[:, 1]), orbit)
    except (FileNotFoundError, ValueError):
        xlab, x0, x1 = "line", 0, nl
        where = "no OAT file"

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("%s  |  %s  |  %d x %d x %d  |  %s"
                 % (p.product_id, p.level, nb, nl, ns, where), fontsize=11)
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 2], hspace=0.28, wspace=0.25)

    # band images (lines downsampled, rotated so along-track is x)
    for i, wl_nm in enumerate(QL_WAVELENGTHS):
        b = p.band(wl_nm)
        img = np.asarray(c[b, ::LINE_DS, :], dtype=np.float32).T
        ax = fig.add_subplot(gs[0, i])
        v1, v2 = np.percentile(img, (2, 98))
        ax.imshow(img, cmap="gray", vmin=v1, vmax=v2, aspect="auto", extent=[x0, x1, ns, 0])
        ax.set_title("%.0f nm (band %d)" % (p.wl[b], b), fontsize=9)
        ax.set_xlabel(xlab)
        if i == 0:
            ax.set_ylabel("sample")

    # browse PNG if the bundle has one
    axb = fig.add_subplot(gs[0, 3])
    brw = (glob.glob(os.path.join(p.root, "browse", "**", "*.png"), recursive=True)
           if os.path.isdir(p.root) else [])
    if brw:
        im = plt.imread(brw[0])
        axb.imshow(im.T if im.ndim == 2 else np.transpose(im, (1, 0, 2)), aspect="auto")
        axb.set_title("browse product", fontsize=9)
    else:
        axb.axis("off")

    # spectra: spatial median plus bright and dark pixels (subsampled)
    ax = fig.add_subplot(gs[1, 0:2])
    sub = np.asarray(c[:, ::max(nl // 400, 1), ::4], dtype=np.float32)
    med = np.median(sub, axis=(1, 2))
    b50 = p.band(1555)
    colmed = np.median(sub[b50], axis=0)         # screen dead/saturated columns
    goodcol = (colmed > 0.5 * np.median(colmed)) & (colmed < 0.98 * 65535)
    good_idx = np.flatnonzero(goodcol)
    flat = sub[b50][:, good_idx].ravel()

    def _px(rank):
        r, c_ = np.unravel_index(np.argsort(flat)[rank], (sub.shape[1], len(good_idx)))
        return r, good_idx[c_]

    bright, dark = _px(-20), _px(20)
    ax.plot(p.wl, med, "k", lw=1.5, label="median")
    ax.plot(p.wl, sub[:, bright[0], bright[1]], lw=0.8, label="bright px")
    ax.plot(p.wl, sub[:, dark[0], dark[1]], lw=0.8, label="dark px")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("stored value")
    ax.legend(fontsize=8)
    ax.set_title("spectra", fontsize=9)

    # striping fingerprint: per-column median at 1555 nm
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(np.median(np.asarray(c[b50, ::LINE_DS, :], dtype=np.float32), axis=0), lw=0.7)
    ax.set_xlabel("sample")
    ax.set_ylabel("column median")
    ax.set_title("striping fingerprint at 1555 nm", fontsize=9)

    # histogram at 1555 nm
    ax = fig.add_subplot(gs[1, 3])
    ax.hist(np.asarray(c[b50, ::LINE_DS * 4, :]).ravel(), bins=100)
    ax.set_xlabel("value at 1555 nm")
    ax.set_yscale("log")
    ax.set_title("histogram", fontsize=9)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, p.product_id + "_quicklook.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved ->", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("product", help="bundle directory, .qub or .xml path")
    ap.add_argument("--out", required=True, help="output directory for the PNG")
    a = ap.parse_args()
    quicklook(a.product, a.out)
