import glob
import os
import zipfile
from tqdm import tqdm

def compress_results():

    os.makedirs('compressed_results', exist_ok=True)

    eval_files = []
    eval_files.extend(glob.glob('results/*every_day*/**/evaluation.csv', recursive=True))
    eval_files.extend(glob.glob('results/*10splits*/**/evaluation.csv', recursive=True))
    eval_files.extend(glob.glob('results/*5splits*/**/evaluation.csv', recursive=True))

    total_size = 0
    for eval_file in tqdm(eval_files):
        zip_path = eval_file.replace('results', 'compressed_results') + '.zip'
        if not os.path.exists(zip_path):
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(eval_file, arcname=os.path.basename(eval_file))
        total_size += os.path.getsize(zip_path)

    print(f'Total size of compressed files: {total_size / (1024**3):.2f} GB')


def decompress_results():

    compressed_files = glob.glob('compressed_results/**/*.zip', recursive=True)

    for compressed_file in compressed_files:
        extract_to = os.path.dirname(compressed_file).replace('compressed_results', 'results')
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(compressed_file, 'r') as zipf:
            zipf.extractall(extract_to)


if __name__=='__main__':
    compress_results()