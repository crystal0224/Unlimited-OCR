#!/usr/bin/env python3
"""Unlimited-OCR runner — PDF/이미지를 Markdown으로 OCR.

Mac/CPU 환경(NVIDIA GPU 없음)을 위해 Baidu 공식 HuggingFace Space
(baidu/Unlimited-OCR, ZeroGPU)의 `/run_ocr` 엔드포인트를 경유한다.

흐름:
  1. 로컬에서 PDF를 페이지 이미지(PNG)로 분해 (PyMuPDF, GPU 불필요)
  2. 각 페이지/이미지를 Space `/run_ocr`로 OCR (원격 ZeroGPU)
  3. 결과 텍스트를 Markdown 합본으로 저장 + stdout 출력

HF 토큰은 선택사항. 없으면 익명 호출(ZeroGPU rate limit 있음),
있으면 환경변수 HF_TOKEN / ~/.cache/huggingface/token 을 자동 사용한다.

서버가 만든 임시 파일을 되받는 `/explode_pdf` 경로는 gradio 6의
allowed_paths 정책상 403이 나므로 의도적으로 쓰지 않는다 (로컬 분해로 대체).

사용:
  python3 ocr.py document.pdf
  python3 ocr.py scan.png -o out.md --mode base
  python3 ocr.py a.pdf b.pdf --pages 1-3 --dpi 300
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

SPACE_ID = "baidu/Unlimited-OCR"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}
DEFAULT_PROMPT = "document parsing."
TOKEN_HELP_URL = "https://huggingface.co/settings/tokens"


class OcrError(Exception):
    """Space 호출 실패 (rate limit, 네트워크, GPU quota 등)."""


def log(msg: str) -> None:
    """진행 로그는 stderr로 (stdout은 OCR 결과 전용)."""
    print(msg, file=sys.stderr, flush=True)


# ── 토큰 ────────────────────────────────────────────────────────────────────
def find_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    cached = Path.home() / ".cache" / "huggingface" / "token"
    if cached.is_file():
        val = cached.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


# ── 페이지 범위 파싱 ("1-3", "2", "1,4,5") ──────────────────────────────────
def parse_pages(spec: str | None) -> set[int] | None:
    """1-based 사용자 입력을 0-based 인덱스 집합으로 변환. None이면 전체."""
    if not spec:
        return None
    wanted: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 1 or hi < lo:
                raise ValueError(f"잘못된 페이지 범위: {chunk}")
            wanted.update(range(lo - 1, hi))
        else:
            n = int(chunk)
            if n < 1:
                raise ValueError(f"잘못된 페이지 번호: {chunk}")
            wanted.add(n - 1)
    return wanted


# ── PDF → PNG (로컬, GPU 불필요) ────────────────────────────────────────────
def pdf_to_pngs(
    pdf_path: Path, dpi: int, pages: set[int] | None, workdir: Path
) -> list[Path]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise OcrError("PyMuPDF가 필요합니다. 설치: pip install pymupdf") from exc

    doc = fitz.open(pdf_path)
    try:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        total = len(doc)
        indices = sorted(i for i in range(total) if pages is None or i in pages)
        out: list[Path] = []
        for i in indices:
            png = workdir / f"{pdf_path.stem}_p{i + 1:04d}.png"
            doc[i].get_pixmap(matrix=mat).save(png)
            out.append(png)
        return out
    finally:
        doc.close()


def collect_inputs(
    paths: list[str], dpi: int, pages: set[int] | None, workdir: Path
) -> list[tuple[str, Path]]:
    """입력들을 (페이지 라벨, 이미지 경로) 목록으로 정규화."""
    items: list[tuple[str, Path]] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_file():
            raise OcrError(f"입력 파일을 찾을 수 없음: {path}")
        ext = path.suffix.lower()
        if ext in PDF_EXTS:
            pngs = pdf_to_pngs(path, dpi, pages, workdir)
            if not pngs:
                raise OcrError(f"PDF에서 렌더된 페이지가 없음 (범위 확인): {path}")
            for j, png in enumerate(pngs, 1):
                items.append((f"{path.name} · p{j}", png))
        elif ext in IMAGE_EXTS:
            items.append((path.name, path))
        else:
            raise OcrError(
                f"지원하지 않는 형식: {ext} ({path.name}). "
                f"PDF 또는 이미지({', '.join(sorted(IMAGE_EXTS))})만 가능."
            )
    return items


# ── 원격 OCR ────────────────────────────────────────────────────────────────
def make_client(token: str | None):
    try:
        from gradio_client import Client
    except ImportError as exc:
        raise OcrError("gradio_client가 필요합니다. 설치: pip install gradio_client") from exc
    log(f"→ Space 연결: {SPACE_ID} ({'HF 토큰 사용' if token else '익명'})")
    kwargs = {"verbose": False}
    if token:
        # gradio_client 2.x는 token=, 1.x는 hf_token= 을 받는다.
        import inspect

        params = inspect.signature(Client.__init__).parameters
        kwargs["token" if "token" in params else "hf_token"] = token
    try:
        return Client(SPACE_ID, **kwargs)
    except Exception as exc:  # noqa: BLE001 — 연결 실패는 모두 OcrError로
        raise OcrError(f"Space 연결 실패: {exc}") from exc


def _looks_like_quota(message: str) -> bool:
    low = message.lower()
    return any(k in low for k in ("quota", "rate limit", "429", "exceeded", "too many"))


def ocr_pages(items: list[tuple[str, Path]], mode: str, prompt: str, token: str | None):
    from gradio_client import handle_file

    client = make_client(token)
    results: list[tuple[str, str]] = []
    total = len(items)
    for n, (label, image) in enumerate(items, 1):
        started = time.time()
        log(f"  [{n}/{total}] OCR {label} …")
        try:
            out = client.predict(
                image_path=handle_file(str(image)),
                mode=mode,
                prompt=prompt,
                api_name="/run_ocr",
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if _looks_like_quota(msg) and not token:
                raise OcrError(
                    "ZeroGPU 무료 할당량 초과로 보입니다.\n"
                    f"  → 무료 HF 토큰을 발급해 재시도하세요: {TOKEN_HELP_URL}\n"
                    "  → export HF_TOKEN=hf_xxx 후 다시 실행 (할당량이 크게 늘어납니다).\n"
                    f"  원본 오류: {msg}"
                ) from exc
            raise OcrError(f"{label} OCR 실패: {msg}") from exc

        text = out.get("text", "") if isinstance(out, dict) else str(out)
        log(f"      ✓ {len(text):,} chars · {time.time() - started:.1f}s")
        results.append((label, text))
    return results


# ── 출력 ────────────────────────────────────────────────────────────────────
def to_markdown(results: list[tuple[str, str]], multi_source: bool) -> str:
    if len(results) == 1 and not multi_source:
        return results[0][1].rstrip() + "\n"
    blocks = [f"<!-- {label} -->\n\n{text.rstrip()}\n" for label, text in results]
    return "\n\n---\n\n".join(blocks) + "\n"


def default_output_path(inputs: list[str]) -> Path:
    first = Path(inputs[0]).expanduser()
    return first.with_suffix(".md")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ocr.py",
        description="Unlimited-OCR: PDF/이미지 → Markdown (HF Space 경유, GPU 불필요)",
    )
    ap.add_argument("inputs", nargs="+", help="PDF 또는 이미지 파일 (여러 개 가능)")
    ap.add_argument("-o", "--output", help="결과 .md 경로 (기본: 첫 입력파일명.md). '-'면 stdout만")
    ap.add_argument(
        "--mode",
        choices=("gundam", "base"),
        default="gundam",
        help="gundam=빠름(기본, ZeroGPU 친화) / base=고정밀(1024px)",
    )
    ap.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help=f"OCR 프롬프트 (기본: {DEFAULT_PROMPT!r})"
    )
    ap.add_argument("--dpi", type=int, default=200, help="PDF 렌더 해상도 (기본 200)")
    ap.add_argument("--pages", help="PDF 페이지 선택 예: '1-3' 또는 '2,5,7' (1-based)")
    ap.add_argument("--quiet", action="store_true", help="OCR 결과를 stdout에 출력하지 않음")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        pages = parse_pages(args.pages)
    except ValueError as exc:
        log(f"✗ {exc}")
        return 2

    token = find_token()
    workdir = Path(tempfile.mkdtemp(prefix="uocr_"))

    try:
        items = collect_inputs(args.inputs, args.dpi, pages, workdir)
        log(f"총 {len(items)}개 페이지/이미지 · mode={args.mode} · dpi={args.dpi}")
        results = ocr_pages(items, args.mode, args.prompt, token)
    except OcrError as exc:
        log(f"✗ {exc}")
        return 1

    multi_source = len({lbl.split(" · ")[0] for lbl, _ in results}) > 1
    markdown = to_markdown(results, multi_source)

    if args.output != "-":
        out_path = (
            Path(args.output).expanduser()
            if args.output
            else default_output_path(args.inputs)
        )
        out_path.write_text(markdown, encoding="utf-8")
        total_chars = sum(len(t) for _, t in results)
        log(f"✓ 저장: {out_path}  ({total_chars:,} chars, {len(results)} pages)")

    if not args.quiet:
        sys.stdout.write(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
