"""Validate slide counts, text geometry, LaTeX diagnostics and rendered pages."""

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
BUILD = HERE / '_build'
PDF = BUILD / 'review.pdf'


def run(*arguments):
    return subprocess.run(arguments, capture_output=True, text=True, encoding='utf-8',
                          errors='replace', check=True, timeout=240).stdout


def main():
    source = (HERE / 'review.tex').read_text(encoding='utf-8')
    introduction, vq4dit, dpq = re.split(r'\\section\{(?:VQ4DiT|DPQ)\}', source)
    counts = [len(re.findall(r'\\begin\{frame\}', part)) for part in (introduction, vq4dit, dpq)]
    assert counts == [8, 10, 10], counts
    notes_source = (HERE / 'speaker_notes.tex').read_text(encoding='utf-8')
    assert [int(n) for n in re.findall(r'\\slide\{(\d+)\}', notes_source)] == list(range(1, 29))
    personal = re.compile(r'\b(?:мы|нас|нам|нами|наш[а-яё]*|рассмотрим|проверим|сравним|'
                          r'выбираем|сохраняем|обучаем|квантуем|используем|получаем)\b', re.I)
    assert not personal.search(source), 'Personal or plural first-person wording'
    assert not personal.search(notes_source), 'Personal wording in speaker notes'
    terminology = re.compile(r'\b(?:словар[а-яё]*|книг[а-яё]*)\b', re.I)
    assert not terminology.search(source), 'Use codebook terminology in slides'
    assert not terminology.search(notes_source), 'Use codebook terminology in notes'
    for forbidden in ('JobB', 'RVQ', 'резервный слайд'):
        assert forbidden not in source, forbidden
    log = (BUILD / 'review.log').read_text(encoding='utf-8', errors='replace')
    for message in ('Overfull', 'Undefined control sequence', 'Missing character'):
        assert message not in log, message
    assert not re.search(r'^!', log, re.M), 'LaTeX compilation error'
    info = run('pdfinfo', str(PDF))
    pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
    assert pages == sum(counts) == 28, pages
    for concept in ('ODE', 'SDE', 'DDPM', 'DDIM',
                    'Калибровка и оценка: что запускается',
                    'На что влияет выбор способа генерации',
                    'Обратная генерация не нужна', 'DDPM50'):
        assert concept in introduction, concept
    bbox = BUILD / 'text_boxes.html'
    run('pdftotext', '-bbox-layout', '-enc', 'UTF-8', str(PDF), str(bbox))
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', bbox.read_text(encoding='utf-8'))
    document = ET.fromstring(text)
    namespace = {'p': 'http://www.w3.org/1999/xhtml'}
    records = []
    for index, page in enumerate(document.findall('.//p:page', namespace), 1):
        width, height = float(page.attrib['width']), float(page.attrib['height'])
        words = page.findall('.//p:word', namespace)
        assert len(words) > 15, ('Empty page', index)
        bottom = 0.
        for word in words:
            x1, y1, x2, y2 = [float(word.attrib[key]) for key in ('xMin', 'yMin', 'xMax', 'yMax')]
            assert x1 >= 0 and y1 >= 0 and x2 <= width + .2 and y2 <= height + .2, (index, word.text)
            bottom = max(bottom, y2)
        records.append(dict(slide=index, words=len(words), bottom=bottom, page_height=height))
    assert len(records) == pages
    run('pdftoppm', '-png', '-scale-to', '1400', str(PDF), str(BUILD / 'slide'))
    images = [BUILD / f'slide-{index:02d}.png' for index in range(1, pages + 1)]
    assert all(path.exists() for path in images)
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 18)
    sheets = []
    for start in range(0, pages, 8):
        sheet = Image.new('RGB', (1160, 1420), '#e7e7e7')
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(images[start:start + 8]):
            slide = Image.open(path).convert('RGB')
            assert float((np.asarray(slide.convert('L')) < 220).mean()) > .008, path.name
            slide.thumbnail((560, 315), Image.Resampling.LANCZOS)
            x, y = 10 + offset % 2 * 580, 15 + offset // 2 * 350
            sheet.paste(slide, (x, y))
            draw.text((x + 5, y + 318), str(start + offset + 1), font=font, fill='#333333')
        path = BUILD / f'contact_{start // 8 + 1}.png'
        sheet.save(path)
        sheets.append(path.name)
    shutil.copy2(PDF, HERE / 'review.pdf')
    notes_pdf = BUILD / 'speaker_notes.pdf'
    notes_info = run('pdfinfo', str(notes_pdf))
    notes_pages = int(re.search(r'Pages:\s+(\d+)', notes_info).group(1))
    assert notes_pages == 28, ('Speaker-note pages', notes_pages)
    notes_log = (BUILD / 'speaker_notes.log').read_text(encoding='utf-8', errors='replace')
    for message in ('Overfull', 'Undefined control sequence', 'Missing character'):
        assert message not in notes_log, ('Speaker notes', message)
    notes_bbox = BUILD / 'notes_boxes.html'
    run('pdftotext', '-bbox-layout', '-enc', 'UTF-8', str(notes_pdf), str(notes_bbox))
    notes_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', notes_bbox.read_text(encoding='utf-8'))
    notes_records = []
    for index, page in enumerate(ET.fromstring(notes_text).findall('.//p:page', namespace), 1):
        width, height = float(page.attrib['width']), float(page.attrib['height'])
        words = page.findall('.//p:word', namespace)
        assert len(words) >= 150, ('Insufficient speaker notes', index, len(words))
        for word in words:
            x1, y1, x2, y2 = [float(word.attrib[key]) for key in ('xMin', 'yMin', 'xMax', 'yMax')]
            assert x1 >= 0 and y1 >= 0 and x2 <= width+.2 and y2 <= height+.2, ('Notes', index, word.text)
        notes_records.append(dict(slide=index, words=len(words)))
    shutil.copy2(notes_pdf, HERE / 'speaker_notes.pdf')
    report = dict(validated=True, slides=pages, section_counts=counts, tex_overflows=0,
                  text_inside_page=True, all_pages_nonblank=True, page_bounds=records,
                  contact_sheets=sheets, pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
                  speaker_notes_pages=notes_pages, notes_geometry='passed', notes_counts=notes_records,
                  notes_sha256=hashlib.sha256(notes_pdf.read_bytes()).hexdigest())
    (HERE / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(slides=pages, sections=counts, speaker_notes_pages=notes_pages,
                          geometry='passed', contact_sheets=sheets)))


if __name__ == '__main__':
    main()
