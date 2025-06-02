import json, os
from teds import TEDS
from raw_html_process import GTPreprocessor, PredPreprocessor

GT_PATH = r"D:/MyWorking/originFinTabNet/fintabnet/FinTabNet_1.0.0_cell_test.jsonl"
ID_PATH = r"D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-PDF_Annotations"
PRED_PATH = r"D:\MyWorking\output"

# TEST_GT = ["<table>", "<tr>", "<td>", "</td>", "<td", " colspan=\"3\"", ">", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td", " colspan=\"3\"", ">", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>", "</table>"]
# TEST_PRED = """<table><thead><th></th><th colspan="3"></th></thead><thead><th></th><th></th><th></th><th></th></thead><thead><th></th><th colspan="3"></th></thead><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr></table>"""

# # === TEST ZONE ===
# gt_processor = GTPreprocessor(TEST_GT)
# gt_html = gt_processor.preprocess()

# pred_processor = PredPreprocessor(TEST_PRED)
# pred_html = pred_processor.preprocess()

# teds = TEDS(structure_only=True)
# score = teds(gt_html, pred_html)
# print(score)

# === FOLDER ZONE ===

def get_pred_html_filename(pdf_filename, table_id):
    '''
    Return .html filename from given PDF filename and table ID.
    If the corresponding .json file or match is not found, return None.
    '''
    json_filename = pdf_filename.replace("/", "_").replace(".pdf", "_tables.json")
    json_file_path = os.path.join(ID_PATH, json_filename)

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            for ann in annotations:
                if table_id == ann["fintabnet_source_table_id"]:
                    no_table = ann["document_table_index"]
                    pred_filename = json_filename.replace("tables", f"table_0_{no_table}").replace(".json", ".html")
                    return pred_filename
    except FileNotFoundError:
        print(f"Annotation file not found: {json_file_path}")
        return None

    # If no matching table_id found
    print(f"No matching annotation for table_id {table_id} in {json_filename}")
    return None
    

def get_pred_html(pred_filename):
    pred_html = os.path.join(PRED_PATH, pred_filename)
    try:
        with open(pred_html, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        print(f"File not found: {pred_html}, skipping.")
        return None
        
scores = []
with open(GT_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            # Init
            data = json.loads(line)
            table_id = data.get('table_id')
            pdf_filename = data.get('filename')
            pred_filename = get_pred_html_filename(pdf_filename, table_id)
            if pred_filename is None:
                continue
            # PRED HTML
            pred_html = get_pred_html(pred_filename)
            if pred_html is None:
                continue
            pred_processor = PredPreprocessor(pred_html)
            pred_html = pred_processor.preprocess()

            # GT HTML
            gt_html_raw = data.get('html', {}).get('structure', {}).get('tokens', [])
            gt_processor = GTPreprocessor(gt_html_raw)
            gt_html = gt_processor.preprocess()
            
            teds = TEDS(structure_only=True)
            score = teds(gt_html, pred_html)
            scores.append(score)
            print(score)

        except json.JSONDecodeError as e:
            print(f"Error decoding line: {e}")

avg_score = sum(scores) / len(scores)
print(f"\nAverage TEDS Score: {avg_score:.4f}")
