from __future__ import annotations
from .pipeline import Pipeline
import pandas as pd


def export_pipeline(pipeline: Pipeline, filename: str,include_data=False,include_features=False):

    """Export a pipeline to a file

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to export
    filename : str
        The filename to export the pipeline to
    """
    if(not include_data):
        pipeline.clear_input()
        pipeline.clear_data()
    if(not include_features):
        pipeline.clear_features()
        
    pd.to_pickle(pipeline, filename)
        
        
        
##TODO add import pipeline function