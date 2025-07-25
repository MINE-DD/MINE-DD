"""
Utility functions for the MINEDD package.

"""
import pickle as pkl
import os
import platform
import pathlib
from paperqa.settings import Settings, AgentSettings, ParsingSettings

def configure_settings(model_type: str,
                        embeddings_model: str,
                        paper_dir: str,
                        chunk_size: int = 2500,
                        overlap: int = 250,
                        ) -> Settings:
    """
    Configure PaperQA settings for use with local LLMs

    Args:
        model_type (str): model to use for answering queries
        embeddings_model (str): model to use for embeddings

    Returns:
        Settings: settings object for PaperQA
    """

    local_llm_config = {
        "model_list": [
            {
                "model_name": model_type,
                "litellm_params": {
                    "model": model_type,
                    # Uncomment if using a local server
                    # "api_base": "http://0.0.0.0:11434",
                },
                "answer": {
                    "evidence_k": 20,
                    "evidence_detailed_citations": True,
                    "evidence_summary_length": "about 100 words",
                    "answer_max_sources": 10,
                    "answer_length": "about 600 words, but can be longer",
                    "max_concurrent_requests": 10,
                    "answer_filter_extra_background": False
                },
                "parsing": {
                    "use_doc_details": True
                },
                "prompts" : {"use_json": False}
            }
        ]
    }

    settings = Settings(
        llm=model_type,
        llm_config=local_llm_config,
        summary_llm=model_type,
        summary_llm_config=local_llm_config,
        paper_directory=paper_dir,
        embedding=embeddings_model,
        agent=AgentSettings(
            agent_llm=model_type,
            agent_llm_config=local_llm_config,
            return_paper_metadata=True
        ),
        parsing=ParsingSettings(
            chunk_size=chunk_size,
            overlap=overlap
        ),
        prompts={"use_json": False} # type: ignore
    )
    return settings


def safely_load_pickle_file(pickled_path):
    if not os.path.exists(pickled_path):
        raise FileNotFoundError(f'File {pickled_path} not found')

    # Windows users do not support PosixPath (whose references are contained in pickle files)
    # so when unpickling the embeddings, Python tries to instantiate these PosixPath objects, but fails
    # since they're not supported

    # one way to fix this is to point the PosixPath point to WindowsPath temporarily during unpickling.
    # from: "https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath"

    system = platform.system()
    file_content = None

    if system == "Windows":
        # Save original PosixPath
        temp = pathlib.PosixPath

        try:
            pathlib.PosixPath = pathlib.WindowsPath  # point to the windows path
            with open(pickled_path, 'rb') as f:
                file_content = pkl.load(f)
        finally:
            # restore the original posixpath
            pathlib.PosixPath = temp
    else:
        with open(pickled_path, 'rb') as f:
            file_content = pkl.load(f)

    return file_content
