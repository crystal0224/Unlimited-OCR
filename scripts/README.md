# Mac/CPU에서 바로 쓰기 (HF Space 경유)

> 이 폴더는 fork(`crystal0224/Unlimited-OCR`)에서 추가한 것입니다.
> 상위 README의 추론 코드는 **NVIDIA GPU + CUDA** 전제라 Mac에서 로컬 실행이 안 됩니다.
> `ocr.py`는 Baidu 공식 HuggingFace Space(`baidu/Unlimited-OCR`, ZeroGPU)의 `/run_ocr`를
> `gradio_client`로 호출하므로 **GPU 없이** PDF/이미지를 OCR할 수 있습니다.

## 설치

```bash
pip install gradio_client pymupdf
# 또는
pip install -r scripts/requirements.txt
```

## 사용

```bash
# PDF 전체 → document.md
python3 scripts/ocr.py document.pdf

# 단일 이미지, 고정밀 모드
python3 scripts/ocr.py scan.png --mode base

# 특정 페이지 범위, 300dpi, 출력 경로 지정
python3 scripts/ocr.py report.pdf --pages 1-5 --dpi 300 -o report.md

# 여러 입력을 하나의 합본으로
python3 scripts/ocr.py a.pdf b.pdf -o merged.md
```

| 옵션 | 의미 |
|------|------|
| `--mode gundam` | 기본·빠름 (640px crop, ZeroGPU 친화) |
| `--mode base`   | 고정밀 (1024px), 표/조밀 레이아웃에 권장 |
| `--pages 1-3` / `2,5,7` | PDF 페이지 선택 (1-based) |
| `--dpi 300`     | PDF 렌더 해상도 (기본 200) |
| `-o out.md`     | 출력 경로 (`-`면 stdout만) |
| `--quiet`       | stdout 출력 생략 |

## HF 토큰 (선택)

익명으로도 동작하지만 ZeroGPU 무료 할당량(IP 기준)이 제한적입니다.
무료 토큰을 넣으면 할당량이 크게 늘어납니다.

```bash
export HF_TOKEN=hf_xxx   # https://huggingface.co/settings/tokens (read 권한)
```
스크립트가 `HF_TOKEN` / `HUGGINGFACE_TOKEN` / `~/.cache/huggingface/token`을 자동 감지합니다.

## 동작 방식

1. 로컬에서 PDF → 페이지 PNG (PyMuPDF, GPU 불필요)
2. 각 페이지를 Space `/run_ocr`로 OCR (원격 ZeroGPU, 페이지당 ~15-20초)
3. 결과를 Markdown 합본으로 저장 + stdout 출력

> 서버측 `/explode_pdf`는 생성 파일 회수가 gradio 6 allowed_paths 정책상 403이라
> 사용하지 않습니다. PDF 분해는 항상 로컬에서 수행합니다.

## 주의

- 입력 페이지 이미지가 **Baidu HF Space로 업로드**됩니다. 대외비 문서는 유의하세요.
- 인터넷 연결이 필요합니다.
- GPU가 있는 환경이라면 상위 README의 로컬 추론(transformers/SGLang)이 더 빠릅니다.
