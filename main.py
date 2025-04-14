from chroma_store import store_report, fetch_last_report

sample_report = """
Competitor A launched AI Interview Coach.
"""

# Store this week's sample report
store_report(sample_report, "week_2025_03_25")

# Fetch the most similar historical report
historical = fetch_last_report()
if historical:
    print("\nMost similar historical report:")
    print(historical)