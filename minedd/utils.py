"""
Utility functions for the MINEDD package.

"""

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
                    "evidence_k": 40,
                    "evidence_detailed_citations": True,
                    "evidence_summary_length": "about 100 words",
                    "answer_max_sources": 10,
                    "answer_length": "about 600 words, but can be longer",
                    "max_concurrent_requests": 10,
                    "answer_filter_extra_background": False
                },
                "parsing": {
                    "use_doc_details": True
                }
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
        )
    )
    return settings
