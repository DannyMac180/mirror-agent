"""
Integration tests for the mem0.ai Python library.

This test suite verifies the integration of the mirror-agent with the mem0.ai API,
ensuring correct functionality for storing and retrieving memories, messages, and documents.

Before running these tests, ensure that the following environment variables are set:
    - MEM0_API_KEY: Your mem0.ai API key
    - MEM0_ORG_ID: Your organization ID
    - MEM0_PROJECT_ID: Your project ID
    - MEM0_USER_ID: A unique identifier for the user (e.g., 'test-user')
    - MEM0_APP_ID: The application ID (default is "mirror-agent")
    - MEM0_RUN_ID: The run ID (default is "test-run")
"""

import os
import unittest
from datetime import datetime

# Set default environment variables for testing
os.environ.update({
    'MEM0_USER_ID': 'test-user',
    'MEM0_APP_ID': 'mirror-agent',
    'MEM0_RUN_ID': 'test-run'
})

from dotenv import load_dotenv
load_dotenv()

# Verify required environment variables
required_vars = ['MEM0_API_KEY', 'MEM0_ORG_ID', 'MEM0_PROJECT_ID']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise ValueError(f"Missing required environment variables: {missing}")

from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema import Document

from src.retrieval_graph.mem0_client import Mem0Memory
from src.retrieval_graph.state import State, reduce_docs, reduce_messages


class TestMem0Integration(unittest.TestCase):
    """Test cases for mem0.ai integration."""

    @classmethod
    def setUpClass(cls):
        """Initialize the Mem0Memory client for testing."""
        cls.mem0 = Mem0Memory()

    def test_document_storage(self):
        """Test adding and retrieving a document using mem0.ai."""
        try:
            # Create a test document
            test_doc = Document(
                page_content="Test document content",
                metadata={"source": "test", "timestamp": datetime.now().isoformat()}
            )

            # Store the document using the reduce_docs helper
            docs = reduce_docs(None, [test_doc])
            self.assertEqual(len(docs), 1)

            # Retrieve documents from mem0.ai
            stored_docs = self.mem0.get_documents()
            self.assertTrue(any(doc.page_content == "Test document content" for doc in stored_docs))
        except Exception as e:
            print("\nDocument Storage Error Details:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise

    def test_message_storage(self):
        """Test adding and retrieving conversation messages using mem0.ai."""
        try:
            # Create test messages
            human_msg = HumanMessage(content="Hello, AI!")
            ai_msg = AIMessage(content="Hello, human!")

            # Store messages using the reduce_messages helper
            messages = []
            messages = reduce_messages(messages, human_msg)
            messages = reduce_messages(messages, ai_msg)

            # Retrieve messages from mem0.ai
            stored_messages = self.mem0.get_messages()
            self.assertTrue(any(msg.content == "Hello, AI!" for msg in stored_messages))
            self.assertTrue(any(msg.content == "Hello, human!" for msg in stored_messages))
        except Exception as e:
            print("\nMessage Storage Error Details:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise

    def test_state_integration(self):
        """Test state management integration with mem0.ai."""
        # Create an initial state instance
        state = State()

        # Add a test query
        state.queries = ["test query"]

        # Add a test document to the state
        test_doc = Document(
            page_content="Test retrieved document",
            metadata={"source": "test"}
        )
        state.retrieved_docs = [test_doc]

        # Convert the state to a dictionary and verify its contents
        state_dict = state.dict()
        self.assertEqual(state_dict["queries"], ["test query"])
        self.assertEqual(
            state_dict["retrieved_docs"][0]["page_content"],
            "Test retrieved document"
        )


if __name__ == '__main__':
    unittest.main()