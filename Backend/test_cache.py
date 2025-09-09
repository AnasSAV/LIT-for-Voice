#!/usr/bin/env python3
"""
Test script for the prediction caching system
"""
import asyncio
import aiohttp
import json
import time

async def test_caching_system():
    """Test the prediction caching system"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Prediction Caching System")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        
        # 1. Check cache stats (should be empty initially)
        print("\n1. Checking initial cache stats...")
        async with session.get(f"{base_url}/predictions/cache/stats") as resp:
            if resp.status == 200:
                stats = await resp.json()
                print(f"   Initial cache entries: {stats['memory_entries']} memory, {stats['file_entries']} file")
            else:
                print(f"   ❌ Failed to get cache stats: {resp.status}")
        
        # 2. Run first batch prediction (should compute and cache)
        print("\n2. Running first batch prediction (wav2vec2 + ravdess)...")
        start_time = time.time()
        
        async with session.get(f"{base_url}/predictions/batch?model=wav2vec2&dataset=ravdess&limit=3") as resp:
            if resp.status == 200:
                result = await resp.json()
                first_duration = time.time() - start_time
                print(f"   ✅ First run completed in {first_duration:.2f}s")
                print(f"   Processed: {result['processed_files']} files")
                print(f"   Cached: {result.get('cached', False)}")
                print(f"   Cache hit: {result.get('cache_hit', False)}")
            else:
                print(f"   ❌ First prediction failed: {resp.status}")
                return
        
        # 3. Check cache stats again (should have entries now)
        print("\n3. Checking cache stats after first run...")
        async with session.get(f"{base_url}/predictions/cache/stats") as resp:
            if resp.status == 200:
                stats = await resp.json()
                print(f"   Cache entries: {stats['memory_entries']} memory, {stats['file_entries']} file")
                print(f"   Cache size: {stats['total_size_mb']} MB")
            else:
                print(f"   ❌ Failed to get cache stats: {resp.status}")
        
        # 4. Run same prediction again (should use cache)
        print("\n4. Running same prediction again (should use cache)...")
        start_time = time.time()
        
        async with session.get(f"{base_url}/predictions/batch?model=wav2vec2&dataset=ravdess&limit=3") as resp:
            if resp.status == 200:
                result = await resp.json()
                second_duration = time.time() - start_time
                print(f"   ✅ Second run completed in {second_duration:.2f}s")
                print(f"   Processed: {result['processed_files']} files")
                print(f"   Cached: {result.get('cached', False)}")
                print(f"   Cache hit: {result.get('cache_hit', False)}")
                print(f"   🚀 Speed improvement: {(first_duration/second_duration):.1f}x faster!")
            else:
                print(f"   ❌ Second prediction failed: {resp.status}")
        
        # 5. List cached combinations
        print("\n5. Listing cached combinations...")
        async with session.get(f"{base_url}/predictions/cache/list") as resp:
            if resp.status == 200:
                data = await resp.json()
                combinations = data.get('cached_combinations', [])
                print(f"   Found {len(combinations)} cached combinations:")
                for combo in combinations:
                    print(f"   - {combo['model']} + {combo['dataset']}: {combo['file_count']} files")
            else:
                print(f"   ❌ Failed to list cache: {resp.status}")
        
        # 6. Test different model (should not use cache)
        print("\n6. Testing different model (whisper-base + common-voice)...")
        start_time = time.time()
        
        async with session.get(f"{base_url}/predictions/batch?model=whisper-base&dataset=common-voice&limit=2") as resp:
            if resp.status == 200:
                result = await resp.json()
                third_duration = time.time() - start_time
                print(f"   ✅ Different model run completed in {third_duration:.2f}s")
                print(f"   Processed: {result['processed_files']} files")
                print(f"   Cached: {result.get('cached', False)}")
                print(f"   Cache hit: {result.get('cache_hit', False)}")
            else:
                print(f"   ❌ Different model prediction failed: {resp.status}")
        
        # 7. Clear specific cache
        print("\n7. Clearing cache for wav2vec2...")
        async with session.delete(f"{base_url}/predictions/cache/clear?model=wav2vec2") as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"   ✅ Cleared {result['invalidated_entries']} cache entries")
            else:
                print(f"   ❌ Failed to clear cache: {resp.status}")
        
        # 8. Final cache stats
        print("\n8. Final cache stats...")
        async with session.get(f"{base_url}/predictions/cache/stats") as resp:
            if resp.status == 200:
                stats = await resp.json()
                print(f"   Final cache entries: {stats['memory_entries']} memory, {stats['file_entries']} file")
            else:
                print(f"   ❌ Failed to get final cache stats: {resp.status}")
    
    print("\n🎉 Cache testing completed!")

if __name__ == "__main__":
    try:
        asyncio.run(test_caching_system())
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
