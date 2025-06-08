import fitz
import matplotlib.pyplot as plt
import json
import numpy as np
import re

class PDFCellExtractor:
    def __init__(self, pdf_path, json_pred_path, json_gt_path, table_number, dpi = 300):    
        self.pdf_path = pdf_path
        self.json_pred_path = json_pred_path
        self.json_gt_path = json_gt_path
        self.table_number = table_number
        self.dpi = dpi
        self.annotations = self._load_annotations()
        # Open PDF document
        self.doc = fitz.open(pdf_path)
        self.page = self.doc[0]

    # ===== GET INFO =====
    # Load JSON annotations from file
    def _load_annotations(self):
        with open(self.json_pred_path, "r") as f:
            annotations = json.load(f)
        return annotations
    
    def get_image_table_bbox(self):
        cell_bboxes = [ann["bbox"] for ann in self.annotations]
        x0 = min(b[0] for b in cell_bboxes)
        y0 = min(b[1] for b in cell_bboxes)
        x1 = max(b[2] for b in cell_bboxes)
        y1 = max(b[3] for b in cell_bboxes)
        return [x0, y0, x1, y1]
    
    def get_pdf_table_bbox(self, table_number):
        with open(self.json_gt_path, "r", encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            pdf_table_bbox = data[table_number]["pdf_table_bbox"]
        return pdf_table_bbox
    
    # ===== LOGIC =====
    def map_cells_to_pdf(self, cell_bbox, image_table_bbox, pdf_table_bbox):
        """
        Map the bounding box of a cell from the image space to the PDF space.
        :param cell_bbox: Bounding box in image space [x0, y0, x1, y1].
        :param image_table_bbox: Bounding box of the table in the image space [x0, y0, x1, y1].
        :return: Mapped bounding box in PDF space [x0_pdf, y0_pdf, x1_pdf, y1_pdf].
        """
        x0_cell, y0_cell, x1_cell, y1_cell = cell_bbox
        x0_img, y0_img, x1_img, y1_img = image_table_bbox
        x0_pdf, y0_pdf, x1_pdf, y1_pdf = pdf_table_bbox
        img_w, img_h = x1_img - x0_img, y1_img - y0_img
        pdf_w, pdf_h = x1_pdf - x0_pdf, y1_pdf - y0_pdf
        # Normalize image coordinates to the range [0, 1]
        x0_norm = (x0_cell - x0_img) / img_w
        y0_norm = (y0_cell - y0_img) / img_h
        x1_norm = (x1_cell - x0_img) / img_w
        y1_norm = (y1_cell - y0_img) / img_h
        # Map normalized coordinates to PDF coordinates
        x0_pdf_mapped = x0_pdf + x0_norm * pdf_w
        y0_pdf_mapped = y0_pdf + y0_norm * pdf_h
        x1_pdf_mapped = x0_pdf + x1_norm * pdf_w
        y1_pdf_mapped = y0_pdf + y1_norm * pdf_h
        return [x0_pdf_mapped, y0_pdf_mapped, x1_pdf_mapped, y1_pdf_mapped]

    def extract_cell_text(self, cell_bbox, image_table_bbox, pdf_table_bbox):
        # Map the cell's bounding box to the PDF space
        cell_pdf_box = self.map_cells_to_pdf(cell_bbox, image_table_bbox, pdf_table_bbox)
        rect = fitz.Rect(cell_pdf_box)
        
        # Extract text from the mapped cell region
        raw_text = self.page.get_text("text", clip=rect).strip()
        cleaned_text = self.clean_cell_text(raw_text)
        return cleaned_text, cell_pdf_box
    
    # ===== POST-PROCESS =====
    def clean_cell_text(self, text):
        """Clean and preprocess cell text."""
        text = re.sub(r'(\.\s*){2,}', ' ', text)  # Remove sequences of dots
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
        return text.strip()
    
    # ===== IMPLEMENTATION =====
    def extract_text(self):
        image_table_bbox = self.get_image_table_bbox()
        pdf_table_bbox = self.get_pdf_table_bbox(self.table_number)
        extracted_cells = []

        for idx, ann in enumerate(self.annotations):
            cell_bbox = ann["bbox"]
            column_nums = ann["column_nums"]
            row_nums = ann["row_nums"]
            cleaned_text, cell_pdf_box = self.extract_cell_text(cell_bbox, image_table_bbox, pdf_table_bbox)
            # print(f"Column nums: {column_nums}")
            # print(f"Row nums: {row_nums}")
            # print(f"Text: {cleaned_text}")
            # print(f"PDF Bbox: {cell_pdf_box}")
            extracted_cells.append({
                "text": cleaned_text,
                "bbox": cell_pdf_box
            })
        return extracted_cells
    
# TEST ZONE
# pdf_path = r"D:\MyWorking\originFinTabNet\fintabnet\pdf\ADS\2007\page_97.pdf"
# json_pred_path = r"D:\MyWorking\output\ADS_2007_page_97_table_0_0_objects.json"
# json_gt_path = r"D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-PDF_Annotations\ADS_2007_page_97_tables.json"

# extractor = PDFCellExtractor(pdf_path, json_pred_path, json_gt_path, 0)
# extractor.extract_text()

class GTCellExtractor:
    def __init__(self, gt_path, table_number):
        self.gt_path = gt_path
        self.table_number = table_number

    def extract_text(self):
        with open(self.gt_path, "r", encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            table = data[self.table_number]
            cell_infos = [
                {
                    "text": cell["json_text_content"],
                    "bbox": cell["pdf_bbox"]
                }
                for cell in table["cells"]
            ]
        return cell_infos