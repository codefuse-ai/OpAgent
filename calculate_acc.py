import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

SITE_PORT_MAP = {
    "7770": "shopping",
    "7780": "shopping_admin",
    "9999": "reddit",
    "8023": "gitlab",
    "8888": "wikipedia",
    "3000": "map",
}

def extract_port_from_url(url: str) -> str:
    """
    Extract port number from URL
    
    Args:
        url: URL string, e.g., 'http://118.31.239.198:7780/admin/dashboard'
    
    Returns:
        Port number as string, or 'unknown' if not found
    """
    # Match pattern like :7780 or :8888 in URL
    match = re.search(r':(\d+)/', url)
    if match:
        port = match.group(1)
        if port in SITE_PORT_MAP:
            return port
    return "unknown"

def calculate_accuracy(results_dir: str) -> Dict:
    """
    Calculate accuracy for all samples in the specified directory
    
    Args:
        results_dir: Path to results directory, e.g., '/path/to/final_visual_results'
    
    Returns:
        Dictionary containing statistics, including:
        - total_samples: Total number of samples
        - successful_samples: Number of successful samples (score > 0)
        - accuracy: Accuracy rate
        - average_score: Average score
        - score_distribution: Score distribution
        - failed_samples: List of failed samples
        - site_statistics: Statistics by website
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Directory not found: {results_dir}")
    
    total_samples = 0
    successful_samples = 0
    total_score = 0.0
    scores = []
    failed_samples = []
    sample_details = []
    
    # Statistics by site
    site_stats = {}
    
    # Iterate through all subdirectories or JSON files
    for item in sorted(results_dir.iterdir()):
        # Handle two directory structures:
        # 1. Subdirectories with trajectory.json inside (e.g., val_0/trajectory.json)
        # 2. Direct JSON files (e.g., 0.json)
        
        if item.is_dir():
            # Structure 1: subdirectory with trajectory.json
            trajectory_file = item / "trajectory.json"
            sample_name = item.name
            
            if not trajectory_file.exists():
                print(f"Warning: trajectory.json not found in {item.name}")
                continue
        elif item.suffix == '.json':
            # Structure 2: direct JSON file
            trajectory_file = item
            sample_name = item.stem  # filename without .json extension
        else:
            continue
        
        try:
            # Read trajectory JSON file
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                trajectory_data = json.load(f)
            
            # Find evaluation result and site information
            score = None
            site_name = "unknown"
            
            for item in trajectory_data:
                if isinstance(item, dict):
                    # Extract score from evaluation
                    if item.get("type") == "evaluation":
                        score = item.get("score", 0.0)
                    # Extract site from configs
                    if "configs" in item and "sites" in item["configs"]:
                        sites = item["configs"]["sites"]
                        if sites and len(sites) > 0:
                            site_name = sites[0]  # Take the first site
            
            if score is None:
                print(f"Warning: Evaluation score not found in {sample_name}")
                continue
            
            # Overall statistics
            total_samples += 1
            total_score += score
            scores.append(score)
            
            sample_info = {
                "sample_name": sample_name,
                "score": score,
                "site": site_name
            }
            sample_details.append(sample_info)
            
            if score > 0:
                successful_samples += 1
            else:
                failed_samples.append(sample_name)
            
            # Site-specific statistics
            if site_name != "unknown":
                # Initialize site stats if not exists
                if site_name not in site_stats:
                    site_stats[site_name] = {
                        "total": 0,
                        "successful": 0,
                        "total_score": 0.0,
                        "scores": [],
                        "failed_samples": []
                    }
                
                site_stats[site_name]["total"] += 1
                site_stats[site_name]["total_score"] += score
                site_stats[site_name]["scores"].append(score)
                if score > 0:
                    site_stats[site_name]["successful"] += 1
                else:
                    site_stats[site_name]["failed_samples"].append(sample_name)
                
        except json.JSONDecodeError as e:
            print(f"Error: Cannot parse {trajectory_file}: {e}")
            continue
        except Exception as e:
            print(f"Error: Error processing {sample_name}: {e}")
            continue
    
    # Calculate overall statistics
    accuracy = (successful_samples / total_samples * 100) if total_samples > 0 else 0.0
    average_score = (total_score / total_samples) if total_samples > 0 else 0.0
    
    # Calculate score distribution
    score_distribution = {}
    for score in scores:
        score_key = f"{score:.1f}"
        score_distribution[score_key] = score_distribution.get(score_key, 0) + 1
    
    # Calculate site-specific statistics
    site_statistics = {}
    for site_name, stats in site_stats.items():
        if stats["total"] > 0:
            # Find port for this site
            port = "N/A"
            for p, s in SITE_PORT_MAP.items():
                if s == site_name:
                    port = p
                    break
            
            site_statistics[site_name] = {
                "port": port,
                "total_samples": stats["total"],
                "successful_samples": stats["successful"],
                "failed_samples_count": stats["total"] - stats["successful"],
                "accuracy": round((stats["successful"] / stats["total"] * 100), 2),
                "average_score": round((stats["total_score"] / stats["total"]), 4),
                "failed_samples": stats["failed_samples"][:5]  # Show first 5 failed samples
            }
    
    results = {
        "total_samples": total_samples,
        "successful_samples": successful_samples,
        "failed_samples_count": len(failed_samples),
        "accuracy": round(accuracy, 2),
        "average_score": round(average_score, 4),
        "score_distribution": dict(sorted(score_distribution.items())),
        "failed_samples": failed_samples[:10],  # Show first 10 failed samples
        "sample_details": sample_details[:5],  # Show first 5 sample details as examples
        "site_statistics": site_statistics
    }
    
    return results


def print_accuracy_report(results_dir: str):
    """
    Print formatted accuracy report
    
    Args:
        results_dir: Path to results directory
    """
    results = calculate_accuracy(results_dir)
    
    print("=" * 60)
    print("Sample Accuracy Statistics Report")
    print("=" * 60)
    print(f"Results Directory: {results_dir}")
    print("-" * 60)
    print(f"Total Samples: {results['total_samples']}")
    print(f"Successful Samples: {results['successful_samples']}")
    print(f"Failed Samples: {results['failed_samples_count']}")
    print(f"Accuracy: {results['accuracy']}%")
    print(f"Accuracy for 812 samples: {results['successful_samples']/812.0:.4f} ({results['successful_samples']}/812)")
    print(f"Average Score: {results['average_score']}")
    print("-" * 60)
    print("Score Distribution:")
    for score, count in results['score_distribution'].items():
        percentage = (count / results['total_samples'] * 100)
        print(f"  Score {score}: {count} samples ({percentage:.2f}%)")
    print("-" * 60)
    
    # Print site-specific statistics
    if 'site_statistics' in results and results['site_statistics']:
        print("Statistics by Website:")
        print("-" * 60)
        for site_name, stats in sorted(results['site_statistics'].items()):
            print(f"\n{site_name.upper()} (Port: {stats['port']}):")
            print(f"  Total Samples: {stats['total_samples']}")
            print(f"  Successful Samples: {stats['successful_samples']}")
            print(f"  Failed Samples: {stats['failed_samples_count']}")
            print(f"  Accuracy: {stats['accuracy']}%")
            print(f"  Average Score: {stats['average_score']}")
            if stats['failed_samples']:
                print(f"  Failed Sample Examples: {', '.join(stats['failed_samples'][:3])}")
        print("-" * 60)
    
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Process final_visual_results directory
    visual_results_dir = "./webarena_results/final_visual_results"
    
    print("\n" + "=" * 80)
    print("PROCESSING FINAL_VISUAL_RESULTS DIRECTORY")
    print("=" * 80 + "\n")
    
    try:
        if os.path.exists(visual_results_dir):
            results = print_accuracy_report(visual_results_dir)
            
            # Save results to file
            output_file = Path(visual_results_dir).parent / "accuracy_report_visual.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to: {output_file}")
        else:
            print(f"Directory not found: {visual_results_dir}")
    except Exception as e:
        print(f"Error processing visual_results: {e}")
    
    # Process final_results directory
    print("\n\n" + "=" * 80)
    print("PROCESSING FINAL_RESULTS DIRECTORY")
    print("=" * 80 + "\n")
    
    final_results_dir = './webarena_results/final_results'
    
    try:
        if os.path.exists(final_results_dir):
            results = print_accuracy_report(final_results_dir)
            
            # Save results to file
            output_file = Path(final_results_dir).parent / "accuracy_report_final.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nDetailed results saved to: {output_file}")
        else:
            print(f"Directory not found: {final_results_dir}")
    except Exception as e:
        print(f"Error processing final_results: {e}")

