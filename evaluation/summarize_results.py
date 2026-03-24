import os
import json
import glob
import pandas as pd

def shorten_model_name(model_path):
    parts = model_path.rstrip('/').split('/')
    if 'checkpoint' in parts[-1]:
        # e.g. mem_factory_qwen3_1.7B-noshuffle/checkpoint_250
        parent = parts[-2]
        ckpt = parts[-1].replace('checkpoint_', '')
        
        # simplify parent name
        parent = parent.replace('mem_factory_', '')
        parent = parent.replace('qwen3_', '')
        
        return f"{parent}_{ckpt}"
    else:
        return parts[-1]

def shorten_dataset_name(dataset_path):
    return os.path.basename(dataset_path).replace('.json', '')

def main():
    results_dir = '/home/guozl/project/MemRL/Memory-CookBook/evaluation/eval_results'
    output_md = os.path.join(results_dir, 'summary.md')
    output_csv = os.path.join(results_dir, 'summary.csv')
    
    files = glob.glob(os.path.join(results_dir, '*.json'))
    
    data = []
    for f in files:
        # We only need to read the summary part, but files are small enough for json.load
        try:
            with open(f, 'r', encoding='utf-8') as file:
                # Read line by line until we have the summary block to avoid loading large results
                # Actually, json.load might take a couple of seconds for 3MB, let's just use it
                content = json.load(file)
                summary = content.get('summary')
                if not summary:
                    continue
                
                model_raw = summary.get('model', 'unknown')
                dataset_raw = summary.get('dataset', 'unknown')
                accuracy = summary.get('current_accuracy', 0.0)
                
                model_short = shorten_model_name(model_raw)
                dataset_short = shorten_dataset_name(dataset_raw)
                
                data.append({
                    'Model': model_short,
                    'Dataset': dataset_short,
                    'Accuracy': accuracy
                })
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    if not data:
        print("No valid data found.")
        return

    df = pd.DataFrame(data)
    
    # Pivot table: rows = Model, columns = Dataset, values = Accuracy
    pivot_df = df.pivot(index='Model', columns='Dataset', values='Accuracy')
    
    # Add an Average column
    pivot_df['Average'] = pivot_df.mean(axis=1)
    
    # Format to 4 decimal places
    pivot_df = pivot_df.round(4)
    
    # Save to CSV
    pivot_df.to_csv(output_csv)
    
    # Generate Markdown table manually to avoid tabulate dependency
    columns = ['Model'] + list(pivot_df.columns)
    header = '| ' + ' | '.join(columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    
    md_lines = [header, separator]
    
    for index, row in pivot_df.iterrows():
        row_str = f"| {index} | " + " | ".join([f"{val:.4f}" if pd.notna(val) else "N/A" for val in row]) + " |"
        md_lines.append(row_str)
        
    md_table = '\n'.join(md_lines)
    
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Evaluation Summary\n\n")
        f.write(md_table)
        f.write("\n")
        
    print(f"Summary generated successfully at {output_md} and {output_csv}")
    print("\n" + md_table)

if __name__ == '__main__':
    main()
