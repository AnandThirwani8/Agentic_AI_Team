#!/usr/bin/env python

from utils import *
from smolagents import GradioUI
import glob

def main():
    print("Gathering PDF paths...")
    pdf_paths = glob.glob('./data/*')
    print(f"Found {len(pdf_paths)} PDF(s).")

    print("Creating vector store...")
    data_vector_store = create_vector_store(pdf_paths)
    print("Vector store created successfully.")

    print("Creating agentic team...")
    agent_manager = create_agentic_team(data_vector_store, web_search_required=False)
    print("Agent created successfully.")

    print("Launching UI now...")
    GradioUI(agent_manager).launch()

if __name__ == '__main__':
    main()
