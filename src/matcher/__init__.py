from .bf_matcher import BF_Matcher
from .mnn_matcher import MNN_Matcher
from .superglue_matcher import SuperGlue_Matcher

matcher_list = {
    'bf': BF_Matcher,
    'mnn': MNN_Matcher,
    'superglue': SuperGlue_Matcher
}