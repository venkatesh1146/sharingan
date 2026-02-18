import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.db.mongodb import get_mongodb_client
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Themed data to add to snapshots
THEMED_ITEMS = [
    {
        "sector": "Banking & Financial Services",
        "relevant_companies": ["HDFC", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra Bank"],
        "sentiment": "bullish",
        "sentiment_score": 0.75
    },
    {
        "sector": "Information Technology",
        "relevant_companies": ["TCS", "Infosys", "Wipro", "HCL Tech", "Tech Mahindra"],
        "sentiment": "neutral",
        "sentiment_score": 0.55
    },
    {
        "sector": "Oil & Gas",
        "relevant_companies": ["Reliance", "ONGC", "Oil India", "Hinduja Solar"],
        "sentiment": "bearish",
        "sentiment_score": 0.35
    },
]


async def update_snapshots_with_themes():
    """Update all snapshots with themed items."""
    try:
        # Initialize MongoDB connection
        client = get_mongodb_client()
        await client.connect()
        
        collection = client.get_collection("market_snapshots")
        
        # Get all snapshots - filter only the ones we just inserted
        snapshots = await collection.find({"snapshot_id": {"$regex": "2026-01-31_07"}}).to_list(None)
        
        print(f"\n📊 Found {len(snapshots)} snapshots to update")
        print(f"🎨 Adding {len(THEMED_ITEMS)} themed sectors to each snapshot...\n")
        
        updated_count = 0
        
        for snapshot in snapshots:
            snapshot_id = snapshot.get("snapshot_id")
            
            # Update themed field by replacing the entire document
            snapshot["themed"] = THEMED_ITEMS
            
            result = await collection.replace_one(
                {"snapshot_id": snapshot_id},
                snapshot
            )
            
            updated_count += result.modified_count
            status = "✅" if result.modified_count > 0 else "ℹ️"
            print(f"  {status} {snapshot_id}")
        
        print(f"\n✅ Successfully updated {len(snapshots)} snapshots with themed items")
        
        # Verify by showing a sample
        sample = await collection.find_one({})
        if sample and sample.get("themed"):
            print(f"\n📋 Sample themed item from snapshot:")
            themed_sample = sample["themed"][0]
            print(f"   Sector: {themed_sample['sector']}")
            print(f"   Sentiment: {themed_sample['sentiment']}")
            print(f"   Companies: {', '.join(themed_sample['relevant_companies'][:3])}...")
            print(f"   Score: {themed_sample.get('sentiment_score', 'N/A')}")
        
        print(f"\n📊 Themed items breakdown:")
        for item in THEMED_ITEMS:
            print(f"   • {item['sector']} ({item['sentiment']})")
        
        await client.disconnect()
        
    except Exception as e:
        logger.error(f"❌ Error updating snapshots: {str(e)}")
        print(f"\n❌ Error updating snapshots: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(update_snapshots_with_themes())