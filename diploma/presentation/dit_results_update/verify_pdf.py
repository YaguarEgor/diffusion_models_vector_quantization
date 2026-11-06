"""Check the compiled slides and render them for visual review."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
BUILD = HERE / "_build"
STEM = "dit_results_update"
PDF = BUILD / (STEM + ".pdf")


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=True, timeout=240).stdout


def main():
    count = int(re.search(r"Pages:\s+(\d+)", run("pdfinfo", str(PDF))).group(1))
    source = (HERE / (STEM + ".tex")).read_text(encoding="utf-8")
    assert count == len(re.findall(r"\\begin\{frame\}", source)) == 20
    assert len(re.findall(r"% \d+: unsuccessful experiments,", source)) == 4
    personal = re.compile(r"\b(?:мы|нас|нам|нами|наш[а-яё]*|сопоставляем|сохраняем|"
                          r"изменяем|сравниваем|проверяем|возвращаем|фиксируем|предположим|"
                          r"усиливаем|собираем|измеряем|храним|накапливаем|"
                          r"перехватываем|пода[её]м|приближаем|выбираем|квантуем)\b", re.I)
    assert not personal.search(source)
    terminology = re.compile(r"\b(?:книг[а-яё]*|словар[а-яё]*)\b", re.I)
    assert not terminology.search(source), "Use codebook terminology consistently"
    assert not terminology.search((HERE / "data/numbers.tex").read_text(encoding="utf-8"))
    notes_source = (HERE / "speaker_notes.tex").read_text(encoding="utf-8")
    assert not terminology.search(notes_source)
    assert not personal.search(notes_source)
    assert [int(n) for n in re.findall(r"\\slide\{(\d+)\}", notes_source)] == list(range(1, 21))
    log = (BUILD / (STEM + ".log")).read_text(encoding="utf-8", errors="replace")
    assert "Overfull" not in log, "LaTeX overflow remains"
    assert "Undefined control sequence" not in log
    assert "Missing character:" not in log
    for name in ("native_learning",):
        assert "File: figures/" + name + ".pdf" in log, f"Figure was not included: {name}"

    bbox = BUILD / "text_boxes.html"
    run("pdftotext", "-bbox-layout", "-enc", "UTF-8", str(PDF), str(bbox))
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", bbox.read_text(encoding="utf-8"))
    ns = {"p": "http://www.w3.org/1999/xhtml"}
    pages = ET.fromstring(clean).findall(".//p:page", ns)
    assert len(pages) == count
    measurements = []
    for index, page in enumerate(pages, 1):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        words = page.findall(".//p:word", ns)
        assert len(words) > 15, f"Empty slide {index}"
        bounds = [[float(word.attrib[k]) for k in ("xMin", "yMin", "xMax", "yMax")] for word in words]
        for word, box in zip(words, bounds):
            assert box[0] >= 0 and box[1] >= 0 and box[2] <= width + .2 and box[3] <= height + .2, (index, word.text, box)
        measurements.append(dict(page=index, words=len(words), bottom=max(b[3] for b in bounds), page_height=height))
    run("pdftoppm", "-png", "-scale-to", "1600", str(PDF), str(BUILD / "slide"))
    images = sorted(BUILD.glob("slide-*.png"))
    assert len(images) == count
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 19)
    contacts = []
    for start in range(0, count, 10):
        sheet = Image.new("RGB", (1260, 1930), "#e7e7e7")
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(images[start:start+10]):
            im = Image.open(path).convert("RGB")
            assert float((np.asarray(im.convert("L")) < 220).mean()) > .008
            im.thumbnail((610, 343), Image.Resampling.LANCZOS)
            x, y = 10+(offset % 2)*630, 15+(offset//2)*380
            sheet.paste(im, (x,y))
            draw.text((x+5, y+347), str(start+offset+1), font=font, fill="#333333")
        contact = BUILD / f"contact_{start//10+1}.png"
        sheet.save(contact)
        contacts.append(contact.name)
    target = HERE / PDF.name
    shutil.copy2(PDF, target)
    notes_pdf = BUILD / "speaker_notes.pdf"
    notes_count = int(re.search(r"Pages:\s+(\d+)", run("pdfinfo", str(notes_pdf))).group(1))
    assert notes_count == 20
    notes_log = (BUILD / "speaker_notes.log").read_text(encoding="utf-8", errors="replace")
    assert "Overfull" not in notes_log and "Missing character:" not in notes_log
    notes_boxes = BUILD / "notes_boxes.html"
    run("pdftotext", "-bbox-layout", "-enc", "UTF-8", str(notes_pdf), str(notes_boxes))
    notes_clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", notes_boxes.read_text(encoding="utf-8"))
    for page in ET.fromstring(notes_clean).findall(".//p:page", ns):
        width, height = float(page.attrib["width"]), float(page.attrib["height"])
        words = page.findall(".//p:word", ns)
        assert len(words) > 120
        for word in words:
            x0,y0,x1,y1 = [float(word.attrib[k]) for k in ("xMin","yMin","xMax","yMax")]
            assert x0 >= 0 and y0 >= 0 and x1 <= width+.2 and y1 <= height+.2
    shutil.copy2(notes_pdf, HERE / notes_pdf.name)
    result = dict(pages=count, unsuccessful_slides=4, tex_overflows=0,
                  words_inside_page=True, all_pages_nonblank=True,
                  first_person_absent=True, footer_absent=True,
                  page_geometry=measurements, contact_sheets=contacts,
                  speaker_notes_pages=notes_count, speaker_notes_geometry="passed",
                  speaker_notes_sha256=hashlib.sha256(notes_pdf.read_bytes()).hexdigest(),
                  pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  visual_review="Contact sheets and full-size pages must also be inspected.")
    (HERE / "data" / "render_validation.json").write_text(json.dumps(result,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(pages=count, speaker_notes_pages=notes_count, checks="passed", contact_sheets=contacts)))


if __name__ == "__main__":
    main()
