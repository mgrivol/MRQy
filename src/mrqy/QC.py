# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:12:28 2023

By: Amir R. Sadri (ars329@case.edu)
"""


import datetime
import time
import os
import argparse
from typing import Generator
import yaml
import pydicom 
import numpy as np
from joblib import Parallel, delayed
from itertools import accumulate
from collections import Counter, namedtuple
from skimage.filters import threshold_otsu
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from skimage import exposure as ex
import pandas as pd
import matplotlib.cm as cm
from medpy.io import load
from pathlib import Path
from scipy.io import loadmat
import warnings        
warnings.filterwarnings("ignore")  


from mrqy import metrics


ProgArgs = namedtuple("ProgArgs", [
        "input_dir",
        "output_dir", 
        "save_masks_flag", 
        "sample_size",
        "middle_size",
        "scan_type",
        "max_num_participants",
        "results_file",
        "results_iqm_file",
        "num_threads"
    ]
)
WorkerArgs = namedtuple("WorkerArgs", ["name", "scans", "subject_type", "tag_data", "total_tags"])



def clean_value(number):
    if isinstance(number, (int, float)):
        number = '{:.2f}'.format(number)
        if number.replace(".", "", 1).isdigit():
            number = float(number)
            if number % 1 == 0:
                number = int(number)
            else:
                number = round(number, 2)
    number = np.array(number)
    return number


def extract_tags(inf, tag_data):
    non_tag_value = 'NA'
    pre_tags = pd.DataFrame.from_dict(tag_data, orient='index', columns=['Tag Abbreviation']).reset_index()
    pre_tags = pre_tags.rename(columns={'index': 'Tag Name'})
    pre_tags['Tag Abbreviation'] = pre_tags['Tag Abbreviation'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
    tags = pd.DataFrame(columns=['Tag', 'Value'])
    for i, row in pre_tags.iterrows():
        pre_tag = row['Tag Name']
        pre_tag = pre_tag.replace(" ", "")
        try:
            tag_value = inf.get(pre_tag,non_tag_value)
            tag_value = clean_value(tag_value)
        except KeyError:
            tag_value = non_tag_value
        tag = row['Tag Abbreviation']
        mul_tags = tag.split(',')
        for j,k in enumerate(mul_tags):
            if np.iterable(tag_value):
                tag_value_j = tag_value[j] if j < len(tag_value) else non_tag_value
            else:
                tag_value_j = tag_value
            new_row = {'Tag': k, 'Value': tag_value_j}
            tags = pd.concat([tags, pd.DataFrame([new_row])], ignore_index=True)
    return tags


def find_scans(root, max_num_images):
    files = []

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:

            if not filename.endswith(('.dcm', '.mha', '.nii', '.gz', '.mat')):
                continue

            files.append(str(Path(dirpath) / filename))

            if max_num_images is not None and len(files) == max_num_images:
                return files

    return files


def input_data(root, max_num_images):
    files = find_scans(root, max_num_images)
        
    dicom_files = [i for i in files if i.endswith('.dcm')]
    mha_files = [i for i in files if i.endswith('.mha')]
    nifti_files = [i for i in files if i.endswith('.nii') or i.endswith('.gz')]
    mat_files = [i for i in files if i.endswith('.mat')]

    def extract_subject_id(filename):
        subject_id = Path(filename).stem
        if subject_id.endswith('.nii'):  # For files like .nii.gz
            subject_id = subject_id[:-4]
        return subject_id.split('.')[0]

    mhas_subjects = [extract_subject_id(scan) for scan in mha_files]
    nifti_subjects = [extract_subject_id(scan) for scan in nifti_files]
    mat_subjects = [extract_subject_id(scan) for scan in mat_files]

    dicom_pre_subjects = [pydicom.dcmread(i).PatientID for i in dicom_files] 
    duplicateFrequencies_dicom = Counter(dicom_pre_subjects)
    dicom_subjects = list(duplicateFrequencies_dicom.keys())

    dicom_scan_numbers = list(duplicateFrequencies_dicom.values())
    ind = [0] + list(accumulate(dicom_scan_numbers))
    dicom_splits = [dicom_files[ind[i]:ind[i+1]] for i in range(len(ind)-1)]

    subjects_id = dicom_subjects + mhas_subjects + nifti_subjects + mat_subjects

    data = {'subject_id': subjects_id, 'subject_type': ['dicom' if subject in dicom_subjects else 'mha' if subject in mhas_subjects else 'nifti' if subject in nifti_subjects else 'mat' for subject in subjects_id]}
    df = pd.DataFrame(data)
    df['dicom_splits'] = [dicom_splits[dicom_subjects.index(subject)] if subject in dicom_subjects else None for subject in df['subject_id']]
    df['path'] = [path if subject_type == 'dicom' else
                  next((mha for mha in mha_files if extract_subject_id(mha) == subject), None) if subject_type == 'mha' else
                  next((nifti for nifti in nifti_files if extract_subject_id(nifti) == subject), None) if subject_type == 'nifti' else
                  next((mat for mat in mat_files if extract_subject_id(mat) == subject), None) if subject_type == 'mat' else
                  None
                  for path, subject, subject_type in zip(df['dicom_splits'], df['subject_id'], df['subject_type'])]
    df.drop('dicom_splits', axis=1, inplace=True)

    print(f'The number of participants is {len(df)}.')
    return df

def volume(name, scans, subject_type, tag_data, middle_size = 100):
    volumes = []
    if subject_type == 'dicom':
            # institution = Path(scans[0]).parent.parent.name
            scans = scans[int(0.005 * len(scans) * (100 - middle_size)):int(0.005 * len(scans) * (100 + middle_size))]
            inf = pydicom.dcmread(scans[0])
            tags = extract_tags(inf, tag_data)
            # new_row1 = {'Tag': 'NUM', 'Value': len(scans)}
            # new_row2 = {'Tag': 'INS', 'Value': institution}
            first_row = {'Tag': 'Participant ID', 'Value': f"{name}"}
            # tags = pd.concat([tags, pd.DataFrame([new_row2])], ignore_index=True)
            # tags = pd.concat([tags, pd.DataFrame([new_row1, new_row2])], ignore_index=True)
            tags.loc[-1] = first_row
            tags.index = tags.index + 1
            tags = tags.sort_index()
            tags = tags.set_index('Tag')['Value'].to_dict()
            slices = [pydicom.read_file(s) for s in scans]
            slices.sort(key=lambda x: int(x.InstanceNumber))
            images = np.stack([s.pixel_array for s in slices])
            images = images.astype(np.int64)
            volumes.append((images, tags))
    elif subject_type in ['mha', 'nifti']:
            image_data, image_header = load(scans)
            images = [image_data[:,:,i] for i in range(np.shape(image_data)[2])]
            middle_index = len(images) // 2
            slices_to_include = int(middle_size * 0.01 * len(images) / 2)
            images = images[middle_index - slices_to_include: middle_index + slices_to_include]
            images = np.stack(images, axis=0)
            images = np.transpose(images, (0, 2, 1))
            volumes.append(images)
    elif subject_type == 'mat':
            images = loadmat(scans)['vol']
            middle_index = len(images) // 2
            slices_to_include = int(middle_size * 0.01 * len(images) / 2)
            images = images[middle_index - slices_to_include: middle_index + slices_to_include]
            images = np.transpose(images, (2, 0, 1))
            volumes.append(images)
    return volumes


class IQM(dict):
    def __init__(self,v, wargs: WorkerArgs, output_dir: Path, save_masks: bool, sample_size: int, scan_type: str):#participant, total_tags, msg_buffer):
        dict.__init__(self)
        self["warnings"] = [] 
        self["output"] = []
        self.msg_buffer = []

        self.participant = wargs.name
        self.output_dir = output_dir / self.participant
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if save_masks: 
            maskfolder = self.output_dir / 'foreground_masks' / self.participant
            maskfolder.mkdir(parents=True, exist_ok=True)

        self.addToPrintList(0, "Participant", self.participant, 25)
        for volume_data in v:
            if isinstance(volume_data, tuple) and len(volume_data) == 2:
                total_metrics = wargs.total_tags + len(metrics.MFUNCS) + 2  # + 1 for NUM + 1 for INS
                images = volume_data[0]
                tags = volume_data[1]
                for count,metric in enumerate(tags):
                    if count != 0:
                        value = tags[metric]
                        self.addToPrintList(count, metric, value, total_metrics)
            else:
                total_metrics = len(metrics.MFUNCS) + 1  # + 1 for NUM + 1 for INS
                images = volume_data
                count = 0

        participant_scan_number = int(np.ceil(images.shape[0]/sample_size))
        self["participant_scan_number"] = participant_scan_number 
        self["os_handle"] = images      
        outputs_list = []
        for j in range(0, images.shape[0], sample_size):
            I = images[j,:,:]
            self.save_image(I, j, self.output_dir)
            if scan_type == "CT": 
                I = I - np.min(I)  # Apply intensity adjustment only for CT scans 
            F, B, c, f, b = self.foreground(I)
            if save_masks != False: 
                self.save_image(c, j, maskfolder)
                
            outputs = {}
            for func_name in metrics.MFUNCS:

                func = getattr(metrics, func_name)
                name, measure = func(F, B, c, f, b)

                assert name.lower() == func_name.lower(), f"{name.lower()} != {func_name.lower()}"
                outputs[name] = measure

            outputs_list.append(outputs)

        self.msg_buffer.append(f'The number of {participant_scan_number} scans were saved to {self.output_dir} directory.')

        if save_masks!= False: 
            self.msg_buffer(f'The number of {participant_scan_number} maskes were also saved to {maskfolder} directory.')
        
        self.addToPrintList(1, "Name of Images", os.listdir(self.output_dir), 25)
        count +=1
        self.addToPrintList(count, "NUM", participant_scan_number, total_metrics)
        averages = {}
        for key in outputs_list[0].keys():
            values = [dic[key] for dic in outputs_list]
            averages[key] = np.nanmean(values) 
            count +=1
            self.addToPrintList(count, key, averages[key], total_metrics)


    def save_image(self, im, index, folder):
        filename = f"{self.participant}({index}).png"
        plt.imsave(folder / filename, im, cmap=cm.Greys_r)
    

    def foreground(self,img):
        try:
            h = ex.equalize_hist(img[:,:])*255
            oi = np.zeros_like(img, dtype=np.uint16)
            oi[(img > threshold_otsu(img)) == True] = 1
            oh = np.zeros_like(img, dtype=np.uint16)
            oh[(h > threshold_otsu(h)) == True] = 1
            nm = img.shape[0] * img.shape[1]
            w1 = np.sum(oi)/(nm)
            w2 = np.sum(oh)/(nm)
            ots = np.zeros_like(img, dtype=np.uint16)
            new =( w1 * img) + (w2 * h)
            ots[(new > threshold_otsu(new)) == True] = 1 
            conv_hull = convex_hull_image(ots)
            conv_hull = convex_hull_image(ots)
            ch = np.multiply(conv_hull, 1)
            fore_image = ch * img
            back_image = (1 - ch) * img
        except Exception: 
            fore_image = img.copy()
            back_image = np.zeros_like(img, dtype=np.uint16)
            conv_hull = np.zeros_like(img, dtype=np.uint16)
            ch = np.multiply(conv_hull, 1)

        return fore_image, back_image, conv_hull, img[conv_hull], img[conv_hull==False]
    

    def addToPrintList(self, count, metric, value, total_metrics):
        self[metric] = value
        self["output"].append(metric)
        if metric != 'Name of Images' and  metric != 'Participant':
            self.msg_buffer.append(f'{count}/{total_metrics}) The {metric} of the participant {self.participant} is {value}.')
            

    def get_participant_scan_number(self): 
        return self["participant_scan_number"]


def reports_join(generator: Generator[IQM, None, None], prog_args: ProgArgs, total_participants: int):
    total_scans = 0
    participant_index = 0

    for s in generator:

        participant_index += 1
        total_scans += s.get_participant_scan_number()

        with open(prog_args.results_file, "a") as fp:
            if participant_index == 1:
                fp.write("#dataset:" + "\n")
                fp.write("\t".join(s["output"]) + "\n")
            fp.write("\t".join([str(s[field]) for field in s["output"]]) + "\n")

        print(f'-------------- {participant_index}/{total_participants}, Participant name: {s.participant} --------------')
        for msg in s.msg_buffer:
            print(msg)


def worker_run(prog_args: ProgArgs, wargs: WorkerArgs):
    s = IQM(
        v=volume(wargs.name, wargs.scans, wargs.subject_type, wargs.tag_data), 
        wargs=wargs,
        output_dir=prog_args.output_dir,
        save_masks=prog_args.save_masks_flag,
        sample_size=prog_args.sample_size,
        scan_type=prog_args.scan_type
    )
    return s



def parse_cmdline() -> ProgArgs:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('outputdir',
                        help = "the output dictory path.",
                        type=str)
    parser.add_argument('inputdir',
                        help = "input foldername consists of *.dcm, *.mha, *.nii or *.mat files. For example: 'E:\\Data\\Rectal\\input_data_folder'",
                        type=str)
    parser.add_argument('-s', help="save foreground masks", type=lambda x: False if x == '0' else x, default=False)
    parser.add_argument('-b', help="number of samples", type=int, default=1)
    parser.add_argument('-u', help="percent of middle images", type=int, default=100)
    parser.add_argument('-t', help="type of scan (MRI or CT)", default='MRI', choices=['MRI', 'CT'])
    parser.add_argument('-n', help="maximum number of images to compute metrics", type=int, default=None)
    parser.add_argument('-c', help="number of threads", type=int, default=1)
    args = parser.parse_args() 

    input_dir = Path(args.inputdir).resolve()
    assert input_dir.exists(), input_dir

    output_dir = Path(args.outputdir).resolve()

    rfi = 0
    results_file = output_dir / f"results_{rfi}.tsv"
    while results_file.exists():
        rfi += 1
        results_file = output_dir / f"results_{rfi}.tsv"

    prog_args = ProgArgs(
        input_dir=Path(args.inputdir).resolve(),
        output_dir=Path(args.outputdir).resolve(),
        save_masks_flag=args.s,
        sample_size=args.b,
        middle_size=args.u,
        scan_type=args.t,
        max_num_participants=args.n,
        results_file=results_file,
        results_iqm_file=output_dir / f"IQM_{rfi}.tsv",
        num_threads=args.c
    )

    prog_args.output_dir.mkdir(exist_ok=True, parents=True)

    with open(prog_args.results_file, "w") as fp:
        fp.write(f"#start_time:\t{datetime.datetime.now()}\n")
        fp.write(f"#outdir:\t{prog_args.output_dir}\n")
        fp.write(f"#scantype:\t{prog_args.scan_type}\n")

    return prog_args



def main():
    start_time = time.time() 

    prog_args = parse_cmdline()
    print(f'MRQy for the {prog_args.scan_type} data is starting....')

    df = input_data(prog_args.input_dir, prog_args.max_num_participants)
    
    if 'dicom' in df['subject_type'].values:
        tag_filename = "MRI_TAGS.yaml" if prog_args.scan_type == "MRI" else "CT_TAGS.yaml"
        with open(tag_filename, 'rb') as file:
            tag_data = yaml.safe_load(file)
        total_tags = sum(len(value) if isinstance(value, list) else 1 for value in tag_data.values())
        print(f'For each participant with dicom files, {total_tags} tags will be extracted and {len(metrics.MFUNCS)+2} metrics will be computed.')
    else:
        total_tags = 0
        tag_data = []
        print(f'For each participant with nondicom files {len(metrics.MFUNCS)+1} metrics will be computed.')

    time.sleep(3)

    workers_args = []
    for _, row in df.iterrows():
        workers_args.append(
            WorkerArgs(
                name=row['subject_id'],
                scans=row['path'],
                subject_type=row['subject_type'],
                # TODO: these could be independent for each participant (1 is NIFTI, 2 is DICOM, ...)
                tag_data=tag_data,
                total_tags=total_tags
            )
        )

    res = Parallel(n_jobs=prog_args.num_threads, return_as="generator")(
        delayed(worker_run)(prog_args, wa) for wa in workers_args
    )
    total_scans = reports_join(res, prog_args, len(df))

    cf = pd.read_csv(prog_args.results_file, sep='\t', comment='#')
    cf = cf.drop(['Name of Images'], axis=1)
    cf = cf.fillna('N/A')
    cf.to_csv(prog_args.results_iqm_file, index=False)
    print(f"The IQMs data are saved in the {prog_args.results_iqm_file} file.")
    
    print("Done!")
    print("MRQy backend took", format((time.time() - start_time) / 60, '.2f'),
        "minutes for {} subjects and the overal {} {} scans to run.".format(len(cf), total_scans, prog_args.scan_type))

    

if __name__ == '__main__':
    main()