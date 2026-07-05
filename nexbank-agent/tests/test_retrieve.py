import pytest
from unittest.mock import patch, MagicMock

from kb.retrieve import retrieve, _cache, ChunkResult

# 10 known query / expected category mappings
KNOWN_PAIRS = [
    ("how to activate debit card", "faq"),
    ("what are the maintenance fees for credit card", "faq"),
    ("NexBank personal loan interest rates and yields", "product_terms"),
    ("Eligibility criteria for savings account application", "product_terms"),
    ("payout terms and tenure options for fixed deposit", "product_terms"),
    ("unauthorized transaction customer liability report savings", "dispute_and_complaint"),
    ("How do I handle duplicate charges on my credit card", "dispute_and_complaint"),
    ("Timelines and compensation policies for failed transactions", "regulatory_rbi"),
    ("RBI Ombudsman escalation path for credit card complaints", "regulatory_rbi"),
    ("Fair practices code and customer protection guidelines", "regulatory_rbi")
]


@pytest.mark.parametrize("query,expected_category", KNOWN_PAIRS)
def test_retrieval_relevance_categories(query: str, expected_category: str):
    """Verifies that retrieval queries return chunks matching expected categories."""
    result = retrieve(query, top_k=3)
    assert len(result.chunks) > 0
    # Check that at least one of the top results matches the expected category
    categories = [chunk.category for chunk in result.chunks]
    assert expected_category in categories, f"Expected {expected_category} in {categories} for: {query}"


def test_metadata_filtering_product_type():
    """Verifies that filtering by product_type excludes non-matching documents."""
    # Query something generic but filter specifically on 'savings' product
    result = retrieve(
        query="maintenance fees and activation limits",
        top_k=5,
        product_type="savings"
    )
    
    # Run ingestion verification or check file system lookup to get product type metadata
    from kb.retrieve import get_resources
    collection, _ = get_resources()
    
    # Cross check metadata of retrieved chunks in persistent database
    for chunk in result.chunks:
        # Retrieve the actual document metadata from ChromaDB
        db_res = collection.get(ids=[chunk.id])
        meta = db_res["metadatas"][0]
        assert meta["product_type"] == "savings", (
            f"Expected chunk {chunk.id} to have product_type 'savings', got '{meta['product_type']}'"
        )


def test_metadata_filtering_region():
    """Verifies that filtering by region excludes non-matching documents."""
    result = retrieve(
        query="rbi grievance guidelines and escalation",
        top_k=5,
        region="IN-KA"
    )
    
    from kb.retrieve import get_resources
    collection, _ = get_resources()
    
    for chunk in result.chunks:
        db_res = collection.get(ids=[chunk.id])
        meta = db_res["metadatas"][0]
        assert meta["region"] == "IN-KA", (
            f"Expected chunk {chunk.id} to have region 'IN-KA', got '{meta['region']}'"
        )


def test_retrieval_caching_spy():
    """Confirms that repeating a query pulls results from cache without querying ChromaDB."""
    # Clear cache to avoid side-effects from other tests
    _cache.clear()
    
    # Mock get_resources to return a spy collection
    with patch("kb.retrieve.get_resources") as mock_get_resources:
        mock_collection = MagicMock()
        mock_model = MagicMock()
        
        import numpy as np
        # Mock encoder output to numpy array
        mock_model.encode.return_value = np.array([[0.1] * 384])
        
        # Mock database query output
        mock_collection.query.return_value = {
            "ids": [["kb_doc_mock001"]],
            "documents": [["Mocked content"]],
            "metadatas": [[{
                "id": "kb_doc_mock001",
                "title": "Mock Title",
                "category": "faq",
                "product_type": "savings",
                "region": "IN-national"
            }]],
            "distances": [[0.05]]
        }
        
        mock_get_resources.return_value = (mock_collection, mock_model)
        
        test_query = "unique cache verification query text"
        
        # Turn 1: Cache Miss, should query ChromaDB
        res1 = retrieve(test_query)
        assert not res1.cache_hit
        assert mock_collection.query.call_count == 1
        assert len(res1.chunks) == 1
        
        # Turn 2: Cache Hit, should pull from cache directly without calling query()
        res2 = retrieve(test_query)
        assert res2.cache_hit
        assert mock_collection.query.call_count == 1  # Should still be 1 (cache lookup)
        assert res2.chunks[0].id == "kb_doc_mock001"
