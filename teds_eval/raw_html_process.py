from bs4 import BeautifulSoup

class GTPreprocessor:
    def __init__(self, gt_html):
        self.gt_html = gt_html

    def preprocess(self):
        gt_html = ''.join(self.gt_html)
        gt_html = f"<html>{gt_html}</html>"
        return gt_html
    
class PredPreprocessor:
    def __init__(self, pred_html):
        self.pred_html = pred_html

    def preprocess(self):
        soup = BeautifulSoup(self.pred_html, 'html.parser')
        table = soup.find('table')

        # Replace <th> with <td>
        for th in table.find_all('th'):
            th.name = 'td'

        # Replace <thead> with <tr>
        for thead in table.find_all('thead'):
            thead.name = 'tr'

        new_html = str(table)
        new_html = f"<html>{new_html}</html>"
        return new_html