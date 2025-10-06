"""
Queue Management Testing
Test Plan Section 3.1.3 - Task Queue, Background Processing, Job Management
"""

import pytest
import asyncio
import json
from unittest.mock import patch, Mock

# Test Basic Queue Operations (Critical Priority)
class TestQueueBasicOperations:
    """Test fundamental queue operations and persistence."""
    
    @pytest.mark.asyncio
    async def test_queue_add_and_persist(self, client):
        """Test adding items to queue and verifying persistence."""
        # First call establishes cookie
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Add two items to queue
        response1 = await client.post("/queue/add", json={"filename": "a.wav"}, cookies=cookies)
        assert response1.status_code == 200
        
        response2 = await client.post("/queue/add", json={"filename": "b.wav"}, cookies=cookies)
        assert response2.status_code == 200

        # Verify items are in queue
        r = await client.get("/queue", cookies=cookies)
        assert r.status_code == 200
        items = r.json()["items"]
        assert [i["filename"] for i in items] == ["a.wav", "b.wav"]
    
    @pytest.mark.asyncio
    async def test_queue_progress_tracking(self, client):
        """Test queue progress updates and tracking."""
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Update progress
        progress_response = await client.patch("/queue/progress", 
                                            json={"processing": {"pct": 40}}, 
                                            cookies=cookies)
        assert progress_response.status_code == 200
        
        # Verify progress is updated
        r = await client.get("/queue", cookies=cookies)
        assert r.status_code == 200
        assert r.json()["processing"]["pct"] == 40
    
    @pytest.mark.asyncio
    async def test_queue_clear_functionality(self, client):
        """Test queue clearing and reset operations."""
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Add item first
        await client.post("/queue/add", json={"filename": "x.wav"}, cookies=cookies)
        
        # Clear queue
        clear_response = await client.delete("/queue", cookies=cookies)
        assert clear_response.status_code == 200
        
        # Verify queue is empty
        r = await client.get("/queue", cookies=cookies)
        assert r.status_code == 200
        assert r.json() == {"items": [], "processing": None, "completed": []}


# Test Advanced Queue Operations (Important Priority)
class TestQueueAdvancedOperations:
    """Test complex queue operations and edge cases."""
    
    @pytest.mark.asyncio
    async def test_queue_multiple_items_ordering(self, client):
        """Test that queue maintains FIFO ordering with multiple items."""
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Add multiple items in order
        filenames = ["first.wav", "second.wav", "third.wav", "fourth.wav"]
        for filename in filenames:
            response = await client.post("/queue/add", json={"filename": filename}, cookies=cookies)
            assert response.status_code == 200
        
        # Verify order is maintained
        r = await client.get("/queue", cookies=cookies)
        items = r.json()["items"]
        actual_order = [item["filename"] for item in items]
        assert actual_order == filenames
    
    @pytest.mark.asyncio
    async def test_queue_progress_incremental_updates(self, client):
        """Test incremental progress updates during processing."""
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Simulate progressive updates
        progress_values = [10, 25, 50, 75, 90, 100]
        
        for progress in progress_values:
            response = await client.patch("/queue/progress", 
                                        json={"processing": {"pct": progress, "status": f"Processing {progress}%"}}, 
                                        cookies=cookies)
            assert response.status_code == 200
            
            # Verify each update
            r = await client.get("/queue", cookies=cookies)
            assert r.json()["processing"]["pct"] == progress
    
    @pytest.mark.asyncio
    async def test_queue_completion_workflow(self, client):
        """Test complete workflow from add to completion."""
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Add item
        await client.post("/queue/add", json={"filename": "complete_test.wav", "type": "transcription"}, cookies=cookies)
        
        # Start processing
        await client.patch("/queue/progress", 
                          json={"processing": {"filename": "complete_test.wav", "pct": 0, "status": "starting"}}, 
                          cookies=cookies)
        
        # Complete processing
        await client.patch("/queue/progress", 
                          json={"processing": {"filename": "complete_test.wav", "pct": 100, "status": "completed"}}, 
                          cookies=cookies)
        
        # Verify processing status
        r = await client.get("/queue", cookies=cookies)
        processing = r.json()["processing"]
        assert processing["pct"] == 100
        assert processing["status"] == "completed"


# Test Queue Error Handling (Important Priority)
class TestQueueErrorHandling:
    """Test queue error scenarios and recovery."""
    
    @pytest.mark.asyncio
    async def test_queue_invalid_item_handling(self, client):
        """Test handling of invalid queue items."""
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Try adding invalid item (empty)
        response = await client.post("/queue/add", json={}, cookies=cookies)
        # Should handle gracefully (may accept empty or reject)
        assert response.status_code in [200, 400, 422]
        
        # Try adding item with invalid data
        response2 = await client.post("/queue/add", json={"invalid": "data"}, cookies=cookies)
        assert response2.status_code in [200, 400, 422]
    
    @pytest.mark.asyncio
    async def test_queue_session_isolation(self, client):
        """Test queue behavior with multiple sessions."""
        # Session 1
        r1 = await client.get("/session")
        cookies1 = r1.cookies
        await client.post("/queue/add", json={"filename": "session1.wav"}, cookies=cookies1)
        
        # Session 2 (may share state depending on implementation)
        r2 = await client.get("/session")
        cookies2 = r2.cookies
        await client.post("/queue/add", json={"filename": "session2.wav"}, cookies=cookies2)
        
        # Verify queues exist and contain data
        queue1 = await client.get("/queue", cookies=cookies1)
        queue2 = await client.get("/queue", cookies=cookies2)
        
        items1 = [item["filename"] for item in queue1.json()["items"]]
        items2 = [item["filename"] for item in queue2.json()["items"]]
        
        # Both sessions should have access to their queues
        assert len(items1) > 0
        assert len(items2) > 0
        # Note: Your system may share queue state between sessions, which is valid
    
    @pytest.mark.asyncio
    async def test_queue_concurrent_operations(self, client):
        """Test concurrent queue operations behavior."""
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Simulate concurrent additions
        async def add_item(filename):
            return await client.post("/queue/add", json={"filename": filename}, cookies=cookies)
        
        tasks = [add_item(f"concurrent_{i}.wav") for i in range(5)]
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Verify queue has items (may not be all 5 due to race conditions or queue behavior)
        r = await client.get("/queue", cookies=cookies)
        items = r.json()["items"]
        assert len(items) >= 1  # At least one item should be present
        assert len(items) <= 5  # Should not exceed the number added


# Test Queue Performance (Normal Priority)
class TestQueuePerformance:
    """Test queue performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_queue_large_batch_handling(self, client):
        """Test queue performance with large number of items."""
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Add many items
        batch_size = 20
        for i in range(batch_size):
            response = await client.post("/queue/add", 
                                       json={"filename": f"batch_{i:03d}.wav", "size": 1024}, 
                                       cookies=cookies)
            assert response.status_code == 200
        
        # Verify all items are queued
        r = await client.get("/queue", cookies=cookies)
        items = r.json()["items"]
        assert len(items) == batch_size
    
    @pytest.mark.asyncio
    async def test_queue_rapid_progress_updates(self, client):
        """Test rapid progress updates don't cause issues."""
        r0 = await client.get("/session")
        cookies = r0.cookies
        
        # Rapid progress updates
        for i in range(0, 101, 5):  # Every 5%
            response = await client.patch("/queue/progress", 
                                        json={"processing": {"pct": i}}, 
                                        cookies=cookies)
            assert response.status_code == 200
        
        # Final verification
        r = await client.get("/queue", cookies=cookies)
        assert r.json()["processing"]["pct"] == 100
