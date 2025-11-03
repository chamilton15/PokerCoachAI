#!/usr/bin/env python3
"""
Poker Coach AI - Quick Start Script
Run common analysis tasks easily
"""

import os
import sys

def print_menu():
    print("\n" + "="*70)
    print("                    POKER COACH AI - QUICK START")
    print("="*70)
    print("\n1. Analyze Hero's Session (56 hands)")
    print("2. Extract & Analyze Player from Full Dataset")
    print("3. Show Available Dataset Files")
    print("4. Exit")
    print()

def analyze_hero():
    print("\n🎯 Analyzing Hero's session...")
    print("="*70)
    os.system("python poker_coach.py abs_NLH_handhq_1_Hero_extracted.phhs Hero")

def extract_and_analyze():
    print("\n🔍 Extract & Analyze Player")
    print("="*70)
    print("\nDataset location: /Users/sethfgn/Desktop/DL_Poker_Project/Poker_Data_Set/data/handhq/")
    print("\nExample files:")
    print("  ABS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/abs NLH handhq_1-OBFUSCATED.phhs")
    print("  PS-2009-07-01_2009-07-23_50NLH_OBFU/0.5/...")
    print()
    
    file_path = input("Enter full path to .phhs file (or 'back'): ").strip()
    if file_path.lower() == 'back':
        return
    
    if not os.path.exists(file_path):
        print(f"\n❌ File not found: {file_path}")
        return
    
    player_id = input("Enter player ID (or partial match): ").strip()
    if not player_id:
        print("❌ Player ID required")
        return
    
    friendly_name = input("Enter friendly name (optional, press Enter to skip): ").strip()
    if not friendly_name:
        friendly_name = player_id[:10]
    
    print(f"\n🚀 Extracting and analyzing {friendly_name}...")
    cmd = f'python analyze_any_player.py "{file_path}" "{player_id}" "{friendly_name}"'
    os.system(cmd)

def show_datasets():
    print("\n📁 Available Dataset Files")
    print("="*70)
    
    dataset_path = "/Users/sethfgn/Desktop/DL_Poker_Project/Poker_Data_Set/data/handhq/"
    
    if os.path.exists(dataset_path):
        dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        print(f"\nFound {len(dirs)} dataset directories:\n")
        for dir_name in sorted(dirs)[:10]:  # Show first 10
            print(f"  • {dir_name}")
        
        if len(dirs) > 10:
            print(f"\n  ... and {len(dirs) - 10} more")
        
        print(f"\nFull path: {dataset_path}")
    else:
        print(f"\n❌ Dataset path not found: {dataset_path}")
        print("Please update the path in quickstart.py")
    
    input("\nPress Enter to continue...")

def main():
    while True:
        print_menu()
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            analyze_hero()
        elif choice == '2':
            extract_and_analyze()
        elif choice == '3':
            show_datasets()
        elif choice == '4':
            print("\n👋 Thanks for using Poker Coach AI!")
            break
        else:
            print("\n❌ Invalid choice. Please enter 1-4.")
        
        if choice in ['1', '2']:
            input("\n\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)


