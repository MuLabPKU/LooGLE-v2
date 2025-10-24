import ast
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer


class Evaluator:

    def judge(self, pred_answer, correct_answer, task, source=None):
        if task == "Metric Calculation":
            return self._judge_metric(pred_answer, correct_answer)
        elif task == "Cross-Company Comparison":
            return self._judge_comparison(pred_answer, correct_answer)
        elif task == 'Version Control':
            return self._judge_version_control(pred_answer, correct_answer)
        else:
            return pred_answer == correct_answer

    def _judge_metric(self, pred_answer, correct_answer):
        pred_val = self._normalize_number(pred_answer)
        correct_val = self._normalize_number(correct_answer)

        if pred_val is None or correct_val is None:
            return False

        return abs(pred_val - correct_val) / abs(correct_val) < 0.05

    def _judge_comparison(self, pred_answer, correct_answer):
        try:
            pred_val = self._normalize_number(pred_answer)
            correct_val = self._normalize_number(correct_answer)
            if pred_val is not None and correct_val is not None:
                return abs(pred_val - correct_val) / abs(correct_val) < 0.05
        except:
            pass
        return pred_answer == correct_answer

    def _judge_version_control(self, pred_answer, correct_answer):
        if isinstance(correct_answer, str):
            correct_answer = ast.literal_eval(correct_answer)
        return self._compute_jaccard(correct_answer, pred_answer)

    @staticmethod
    def _normalize_number(s):
        if isinstance(s, list):
            s = s[0] if s else ''
        if not s:
            return None

        try:
            s = s.replace(',', '').replace('$', '').strip()
            multiplier = 1.0

            if s.endswith('M'):
                s = s[:-1]
            elif s.endswith('%'):
                s = s[:-1]
                multiplier = 1

            return float(s) * multiplier
        except ValueError:
            return None

    @staticmethod
    def _compute_jaccard(list1, list2):
        if not list1 or not list2:
            return 0.0

        mlb = MultiLabelBinarizer()
        binarized = mlb.fit_transform([list1, list2])
        score = jaccard_score(binarized[0], binarized[1]) * 100

        return round(score, 2)
