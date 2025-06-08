import os
import json
import re
from rtree import index
import csv
from text_extract import PDFCellExtractor, GTCellExtractor
from bert import FinBERTSimilarity

ID_PATH = r"D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-PDF_Annotations"
PRED_PATH = r"D:\MyWorking\output" # The test folder
PDF_PATH = r"D:\MyWorking\originFinTabNet\fintabnet\pdf"

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / (boxA_area + boxB_area - inter_area)

def fix_bbox_order(bbox):
    """ Ensure the bounding box is in the format [x_min, y_min, x_max, y_max] """
    try:
        x_min, y_min, x_max, y_max = bbox
        # Check if the coordinates are valid
        if x_min > x_max or y_min > y_max:
            return None  # Return None for invalid bounding boxes
        return [x_min, y_min, x_max, y_max]
    except Exception as e:
        print(f"Error with bbox {bbox}: {e}")
        return None  # Return None if any error occurs while processing the bbox

def match_cells(pred_cells, gt_cells, iou_thresh = 0.5):
    gt_index = index.Index()
    for i, gt in enumerate(gt_cells):
        gt_bbox = fix_bbox_order(gt["bbox"]) 
        if gt_bbox is None:  # Skip invalid bounding boxes
            print(f"Skipping invalid GT bbox for cell {i}: {gt['bbox']}")
            continue
        gt_index.insert(i, tuple(gt["bbox"]))

    matched = []
    used_gt_ids = set()

    for pred in pred_cells:
        pred_box = fix_bbox_order(pred["bbox"])  # Fix bbox order if needed
        if pred_box is None:  # Skip invalid bounding boxes
            print(f"Skipping invalid predicted bbox: {pred['bbox']}")
            continue

        # pred_box = pred["bbox"]
        best_iou = 0
        best_gt_id = -1
        for i in gt_index.intersection(tuple(pred_box)):
            if i in used_gt_ids:
                continue
            iou = compute_iou(pred_box, gt_cells[i]["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_id = i

        if best_iou >= iou_thresh:
            matched.append((pred, gt_cells[best_gt_id], best_iou))
            used_gt_ids.add(best_gt_id)

    return matched

# Initialize totals for the overall average score
total_iou = 0
total_similarity = 0
total_matches = 0

# Take the pred text cells
for pred_file in os.listdir(PRED_PATH):
    if pred_file.endswith("_0_objects.json"):
        # Table number
        match = re.search(r'table_(\d+)_', pred_file)
        if match:
            table_number = int(match.group(1))
            
        # 1st parameter
        json_pred_path = os.path.join(PRED_PATH, pred_file)

        # 2nd parameter
        json_gt_path = re.sub(r'(table).*',"tables.json", pred_file)
        json_gt_path = os.path.join(ID_PATH, json_gt_path)

        with open(json_gt_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
            if table_number >= len(data):
                print(f"Warning: table_number {table_number} is out of range for this PDF. Skipping table {pred_file}")
                continue  # Skip this iteration if table_number is invalid

            pdf_folder = data[table_number]["pdf_folder"]
            pdf_filename = data[table_number]["pdf_file_name"]
            pdf_path = pdf_folder + pdf_filename
        # 3rd parameter
            pdf_path = os.path.join(PDF_PATH, pdf_path)
        
        pred_extractor = PDFCellExtractor(pdf_path, json_pred_path, json_gt_path, table_number)
        pred_cells = pred_extractor.extract_text()

        gt_extractor = GTCellExtractor(json_gt_path, table_number)
        gt_cells = gt_extractor.extract_text()
        
        matches = match_cells(pred_cells, gt_cells)

        finbert = FinBERTSimilarity()

        results = []
        cell_iou = 0
        cell_similarity = 0
        cell_matches = 0

        for pred, gt, iou in matches:
            pred_text = pred["text"]
            gt_text = gt["text"]
            similarity_matrix = finbert.compute_similarity([pred_text], [gt_text])
            #print(f"Match: '{pred_text}' ↔ '{gt_text}' | IoU: {iou:.2f} | FinBERT Similarity: {similarity_matrix[0][0]:.2f}")
            similarity_score = similarity_matrix[0][0]

            results.append({
                "pred_text": pred_text,
                "gt_text": gt_text,
                "iou": iou,
                "finbert_similarity": similarity_score
            })

            # Accumulate table scores for average calculation
            cell_iou += iou
            cell_similarity += similarity_score
            cell_matches += 1

        # Calculate average scores for the current table
        if cell_matches > 0:
            avg_iou = cell_iou / cell_matches
            avg_similarity = cell_similarity / cell_matches
        else:
            avg_iou = 0
            avg_similarity = 0
        print(f"Avg IoU: {avg_iou:.2f}, Avg Similarity: {avg_similarity:.2f}, Filename: {pred_file}")

        total_iou += cell_iou
        total_similarity += cell_similarity
        total_matches += cell_matches
        
if total_matches > 0:
    overall_avg_iou = total_iou / total_matches
    overall_avg_similarity = total_similarity / total_matches
else:
    overall_avg_iou = 0
    overall_avg_similarity = 0

print(f"\nOverall Avg IoU: {overall_avg_iou:.2f}")
print(f"Overall Avg Similarity: {overall_avg_similarity:.2f}")

# Overall Avg IoU: 0.69
# Overall Avg Similarity: 0.98