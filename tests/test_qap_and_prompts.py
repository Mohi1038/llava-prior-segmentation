import numpy as np

from llava_prior.config import PriorSegmentationConfig
from llava_prior.prompts import masks_to_prompts
from llava_prior.qap import build_class_query, parse_class_response


def test_qap_query_and_positive_response():
    config = PriorSegmentationConfig()
    query = build_class_query("bird", config.class_prompt)
    assert "class of a bird" in query
    observation = parse_class_response("bird", "Yes, it contains a bird. It is flying.")
    assert observation.present
    assert observation.description == "It is flying"


def test_prompt_extraction_returns_centroid_and_box():
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    prompts = masks_to_prompts(mask, 12, 16, min_component_area=2, morphological_kernel=1)
    assert prompts.points.shape == (1, 2)
    assert np.allclose(prompts.points[0], [4.5, 3.5])
    assert np.array_equal(prompts.box, [3, 2, 6, 5])