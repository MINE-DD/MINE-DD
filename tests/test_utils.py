"""Tests for the minedd.utils module."""
import pytest
from minedd.utils import configure_settings
from paperqa.settings import Settings, AgentSettings


def test_configure_settings_default():
    """Test the configure_settings function with typical inputs."""
    model_type = "ollama/llama3.2:1b"
    embeddings_model = "ollama/mxbai-embed-large:latest"
    paper_dir = "data/"
    
    settings = configure_settings(model_type, embeddings_model, paper_dir)
    
    # Verify return type
    assert isinstance(settings, Settings), "Should return a Settings object"
    
    # Verify model settings are correctly configured
    assert settings.llm == model_type, "LLM model should match input model_type"
    assert settings.embedding == embeddings_model, "Embedding model should match input embeddings_model"
    assert settings.paper_directory == paper_dir, "Paper directory should match input paper_dir"
    
    # Verify agent settings
    assert isinstance(settings.agent, AgentSettings), "Agent settings should be an AgentSettings object"
    assert settings.agent.agent_llm == model_type, "Agent LLM should match input model_type"
    assert settings.agent.return_paper_metadata is True, "Agent should return paper metadata"


def test_configure_settings_custom():
    """Test the configure_settings function with custom model inputs."""
    model_type = "ollama/mistral:latest"
    embeddings_model = "ollama/nomic-embed-text:latest"
    paper_dir = "custom_papers/"
    
    settings = configure_settings(model_type, embeddings_model, paper_dir)
    
    # Verify model settings are correctly configured with custom values
    assert settings.llm == model_type, "LLM model should match custom model_type"
    assert settings.embedding == embeddings_model, "Embedding model should match custom embeddings_model"
    assert settings.paper_directory == paper_dir, "Paper directory should match custom paper_dir"


def test_configure_settings_llm_config():
    """Test that the LLM configuration is properly set up."""
    model_type = "ollama/llama3.2:1b"
    embeddings_model = "ollama/mxbai-embed-large:latest"
    paper_dir = "data/"
    
    settings = configure_settings(model_type, embeddings_model, paper_dir)
    
    # Verify LLM config is correctly structured
    assert settings.llm_config is not None, "LLM config should not be None"
    assert "model_list" in settings.llm_config, "LLM config should contain model_list"
    assert len(settings.llm_config["model_list"]) > 0, "Model list should not be empty"
    
    model_config = settings.llm_config["model_list"][0]
    assert model_config["model_name"] == model_type, "Model name in config should match input model_type"
    assert "litellm_params" in model_config, "Model config should include litellm_params"
    assert model_config["litellm_params"]["model"] == model_type, "Model in litellm_params should match input model_type"
    
    # Verify answer settings
    assert "answer" in model_config, "Model config should include answer settings"
    answer_config = model_config["answer"]
    assert answer_config["evidence_detailed_citations"] is True, "Evidence should include detailed citations"
    assert answer_config["answer_max_sources"] == 10, "Max sources should be set to 10"