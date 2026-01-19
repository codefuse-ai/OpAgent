import json
import logging
from pathlib import Path
from demo.utils import change_mainip2ecsip, extract_answer

logger = logging.getLogger(__name__)

class EvaluatorComb:
    def __init__(self, evaluators):
        self.evaluators = evaluators

    async def __call__(self, solution_str, trajectory, config_file, page):
        score = 1.0
        for evaluator in self.evaluators:
            cur_score = await evaluator(solution_str, trajectory, config_file, page)
            score *= cur_score
        return score

# Mock Evaluators for demo purpose
class StringEvaluator:
    def __init__(self, REPLACE_WITH_YOUR_HOST=None):
        self.REPLACE_WITH_YOUR_HOST = REPLACE_WITH_YOUR_HOST
    async def __call__(self, solution_str, trajectory, config_file, page):
        # Mock evaluation logic
        logger.info("Mock StringEvaluator running")
        return 1.0

class URLExactEvaluator:
    def __init__(self, REPLACE_WITH_YOUR_HOST=None):
        self.REPLACE_WITH_YOUR_HOST = REPLACE_WITH_YOUR_HOST
    async def __call__(self, solution_str, trajectory, config_file, page):
        logger.info("Mock URLExactEvaluator running")
        return 1.0

class HTMLContentExactEvaluator:
    def __init__(self, REPLACE_WITH_YOUR_HOST=None):
        self.REPLACE_WITH_YOUR_HOST = REPLACE_WITH_YOUR_HOST
    async def __call__(self, solution_str, trajectory, config_file, page):
        logger.info("Mock HTMLContentExactEvaluator running")
        return 1.0

class PageImageEvaluator:
    def __init__(self, captioning_fn=None, REPLACE_WITH_YOUR_HOST=None):
        self.REPLACE_WITH_YOUR_HOST = REPLACE_WITH_YOUR_HOST
    async def __call__(self, solution_str, trajectory, config_file, page):
        logger.info("Mock PageImageEvaluator running")
        return 1.0

def evaluator_router(config_file: Path | str, captioning_fn=None, REPLACE_WITH_YOUR_HOST: str = None) -> EvaluatorComb:
    with open(config_file, "r") as f:
        raw = f.read()
        raw = change_mainip2ecsip(raw, REPLACE_WITH_YOUR_HOST) if REPLACE_WITH_YOUR_HOST else raw
        configs = json.loads(raw)

    eval_types = configs["eval"]["eval_types"]
    evaluators = []
    for eval_type in eval_types:
        if eval_type == "string_match":
            evaluators.append(StringEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
        elif eval_type == "url_match":
            evaluators.append(URLExactEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
        elif eval_type == "program_html":
            evaluators.append(HTMLContentExactEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
        elif eval_type == "page_image_query":
            evaluators.append(PageImageEvaluator(captioning_fn, REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST))
        else:
            logger.warning(f"Unknown eval_type: {eval_type}, using dummy pass")
            evaluators.append(StringEvaluator(REPLACE_WITH_YOUR_HOST=REPLACE_WITH_YOUR_HOST)) # Fallback

    return EvaluatorComb(evaluators)
