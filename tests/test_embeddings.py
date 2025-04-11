"""Tests for the minedd.embeddings module."""
import pytest
# import os
# import json
import shutil
from minedd.embeddings import Embeddings
# from paperqa.settings import Settings, AgentSettings

@pytest.fixture()
def embeddings(tmp_path):
    """Fixture to create an Embeddings instance with a temporary directory."""
    temp_dir = tmp_path / "test_embeddings"
    temp_dir.mkdir()
    embeddings = Embeddings(temp_dir/"temp_embeddings.pkl")
    yield embeddings

    # Cleanup: Remove files and directories created during the test
    shutil.rmtree(temp_dir, ignore_errors=True)

# def test_prepare_papers(embeddings):
#     """Test the prepare_papers method."""
#     ordered_files = embeddings.prepare_papers("tests/mock_papers")
#     assert ordered_files == ["nihms-620915.pdf"]


# def test_process_papers(embeddings):
#     """Test the process_papers method."""
#     local_llm_config = json.load(
#         open("minedd/settings_paperqa.json")
#     )
#     model = local_llm_config["model_list"][0]["model_name"]
#     test_papers_dir = "tests/mock_papers"

#     settings = Settings(
#         llm=model,
#         llm_config=local_llm_config,
#         summary_llm=model,
#         summary_llm_config=local_llm_config,
#         paper_directory=test_papers_dir,
#         embedding="snowflake-arctic-embed:22m",
#         agent=AgentSettings(
#             agent_llm=model,
#             agent_llm_config=local_llm_config,
#             return_paper_metadata=True
#         )
#     )
#     embeddings.process_papers(settings, test_papers_dir)

#     # Verify embeddings docs are created
#     assert embeddings.docs is not None

#     # Verify the output file exists
#     assert os.path.exists(embeddings.output_embeddings_path)

