import re


class AnswerExtractor:

    def extract(self, response, task, source=None):
        response = response.replace('*', '')

        if task == 'Legal Case Retrieval':
            return self._extract_pattern(response, r'CASE_\d+')
        elif task == 'Legal Article Extraction':
            return self._extract_pattern(response, r'LAW_\d+')
        elif task == 'Version Control':
            return self._extract_version_control(response)
        elif source == 'Finance' and task == 'Trend Analysis':
            return self._extract_first_match(response, r'\b\d{4}-\d{4}\b')
        elif source == 'Finance':
            return self._extract_first_match(response, r'\d{1,6}(?:,\d{3})*(?:\.\d+)?(?:[M%])?')
        else:
            return self._extract_pattern(response, r'[A-D]')

    def _extract_pattern(self, response, pattern):
        match = re.search(rf'[Tt]he correct answer is[:\s]*[\(<]*({pattern})', response)
        if match:
            return match.group(1)
        return None

    def _extract_first_match(self, response, pattern):
        match = re.search(r"[Tt]he correct answer is[:\s]+(.*)", response)
        if not match:
            return None
        matches = re.findall(pattern, match.group(1))
        return matches[0] if matches else None

    def _extract_version_control(self, response):
        matches = re.findall(r'[\w/]+\.py', response)
        return list(set(matches))
