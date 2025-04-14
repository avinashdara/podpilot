from pinecone_store import store_report, fetch_last_report

sample_report = """
Competitor A launched AI Interview Coach.
Competitor B increased pricing by 20%.
Competitor C launched 2 new data science courses.
"""

# Store this week's sample report
store_report(sample_report, "week_2025_03_25")

# Fetch the most similar historical report
historical = fetch_last_report()
print("Historical Data Retrieved:\n", historical)