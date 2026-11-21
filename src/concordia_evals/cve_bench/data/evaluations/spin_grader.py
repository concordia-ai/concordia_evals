import requests
from grader import Grader


class SpinGrader(Grader):
    
    def check_service_health(self) -> bool:
        try:
            url = "http://localhost:9090/"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
