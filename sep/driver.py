import argparse
import json
import os
import sys

import numpy as np
import pylab
import scipy.io

import utils.img as ig
import utils.imtable as imtable
import utils.sound as sound
import utils.util as ut
import sep_params
import tfutil as mu

from InferenceClient import InferenceClient
from utils.sound import Sound

pj = ut.pjoin

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

# What does this do?
def run(vid_file, start_time, dur, pr, gpu, buf = 0.05, mask = None, arg = None):
  print pr
  dur = dur + buf
  with ut.TmpDir() as vid_path:
    height_s = '-vf "scale=-2:\'min(%d,ih)\'"' % arg.max_full_height if arg.max_full_height > 0 else ''
    ut.sys_check(ut.frm(
      'ffmpeg -loglevel error -ss %(start_time)s -i "%(vid_file)s" -safe 0  '
      '-t %(dur)s -r %(pr.fps)s -vf scale=256:256 "%(vid_path)s/small_%%04d.png"'))
    ut.sys_check(ut.frm(
      'ffmpeg -loglevel error -ss %(start_time)s -i "%(vid_file)s" -safe 0 '
      '-t %(dur)s -r %(pr.fps)s %(height_s)s "%(vid_path)s/full_%%04d.png"'))
    ut.sys_check(ut.frm(
      'ffmpeg -loglevel error -ss %(start_time)s -i "%(vid_file)s" -safe 0  '
      '-t %(dur)s -ar %(pr.samp_sr)s -ac 2 "%(vid_path)s/sound.wav"'))

    if arg.fullres:
      fulls = map(ig.load, sorted(ut.glob(vid_path, 'full_*.png'))[:pr.sampled_frames])
      fulls = np.array(fulls)

    snd = sound.load_sound(pj(vid_path, 'sound.wav'))
    samples_orig = snd.normalized().samples
    samples_orig = samples_orig[:pr.num_samples]
    samples_src = samples_orig.copy()
    if samples_src.shape[0] < pr.num_samples:
      return None
      
    ims = map(ig.load, sorted(ut.glob(vid_path, 'small_*.png')))
    ims = np.array(ims)
    d = 224
    y = x = ims.shape[1]/2 - d/2
    ims = ims[:, y : y + d, x : x + d]
    ims = ims[:pr.sampled_frames]

    if mask == 'l':
      ims[:, :, :ims.shape[2]/2] = 128
      if arg.fullres:
        fulls[:, :, :fulls.shape[2]/2] = 128
    elif mask == 'r':
      ims[:, :, ims.shape[2]/2:] = 128
      if arg.fullres:
        fulls[:, :, fulls.shape[2]/2:] = 128
    elif mask is None:
      pass
    else: raise RuntimeError()

    samples_src = mu.normalize_rms_np(samples_src[None], pr.input_rms)[0]

    ret = InferenceClient.predict(ims[None], samples_src[None])
    samples_pred_fg = ret.samples_pred_fg[0][:, None]
    samples_pred_bg = ret.samples_pred_bg[0][:, None]
    spec_pred_fg = ret.spec_pred_fg[0]
    spec_pred_bg = ret.spec_pred_bg[0]
    print spec_pred_bg.shape
    spec_mix = ret.spec_mix[0]

    if arg.cam:
      cam, vis = find_cam(fulls, samples_orig, arg)
    else:
      if arg.fullres:
        vis = fulls
      else:
        vis = ims

    return dict(ims = vis, 
                samples_pred_fg = samples_pred_fg, 
                samples_pred_bg = samples_pred_bg, 
                samples_mix = ret.samples_mix[0],
                samples_src = samples_src, 
                spec_pred_fg = spec_pred_fg, 
                spec_pred_bg = spec_pred_bg, 
                spec_mix = spec_mix)


def main(args):

  args = objectview(args)

  args.fullres = args.fullres or args.cam

  if args.gpu < 0:
    args.gpu = None

  print 'Start time:', args.start
  print 'GPU =', args.gpu

  gpus = [args.gpu]
  gpus = mu.set_gpus(gpus)
  
  if args.duration_mult is not None:
    pr = sep_params.full()
    step = 0.001 * pr.frame_step_ms
    length = 0.001 * pr.frame_length_ms
    args.clip_dur = length + step*(0.5+pr.spec_len) * args.duration_mult
  
  fn = getattr(sep_params, args.model)
  pr = fn(vid_dur = args.clip_dur)

  if args.clip_dur is None:
    args.clip_dur = pr.vid_dur
  pr.input_rms = np.sqrt(0.1**2 + 0.1**2)
  print 'Spectrogram samples:', pr.spec_len
  pr.model_path = '../results/nets/sep/%s/net.tf-%d' % (pr.name, pr.train_iters)

  if not os.path.exists(args.vid_file):
    print 'Does not exist:', args.vid_file
    sys.exit(1)

  if args.duration is None:
    args.duration = args.clip_dur + 0.01

  print args.duration, args.clip_dur
  full_dur = args.duration
  step_dur = args.clip_dur/2.
  filled = np.zeros(int(np.ceil(full_dur * pr.samp_sr)), 'bool')
  full_samples_fg = np.zeros(filled.shape, 'float32')
  full_samples_bg = np.zeros(filled.shape, 'float32')
  full_samples_src = np.zeros(filled.shape, 'float32')
  args.start = ut.make_mod(args.start, (1./pr.fps))

  ts = np.arange(args.start, args.start + full_dur - args.clip_dur, step_dur)
  full_ims = [None] * int(np.ceil(full_dur * pr.fps))

  # Process each video chunk
  for t in ut.time_est(ts):
    t = ut.make_mod(t, (1./pr.fps))
    frame_start = int(t*pr.fps - args.start*pr.fps)
    ret = run(args.vid_file, t, args.clip_dur, pr, gpus[0], mask = args.mask, arg = args)
    if ret is None:
      continue
    ims = ret['ims']
    for frame, im in zip(xrange(frame_start, frame_start + len(ims)), ims):
      full_ims[frame] = im
    
    samples_fg = ret['samples_pred_fg'][:, 0]
    samples_bg = ret['samples_pred_bg'][:, 0]
    samples_src = ret['samples_src'][:, 0]
    samples_src = samples_src[:samples_bg.shape[0]]

    sample_start = int(round((t - args.start) * pr.samp_sr))
    n = samples_src.shape[0]
    inds = np.arange(sample_start, sample_start + n)
    ok = ~filled[inds]
    full_samples_fg[inds[ok]] = samples_fg[ok]
    full_samples_bg[inds[ok]] = samples_bg[ok]
    full_samples_src[inds[ok]] = samples_src[ok]
    filled[inds] = True

  full_samples_fg = np.clip(full_samples_fg, -1., 1.)
  full_samples_bg = np.clip(full_samples_bg, -1., 1.)
  full_samples_src = np.clip(full_samples_src, -1., 1.)
  full_ims = [x for x in full_ims if x is not None]
  table = [['start =', args.start],
           'fg:', imtable.Video(full_ims, pr.fps, Sound(full_samples_fg, pr.samp_sr)),
           'bg:', imtable.Video(full_ims, pr.fps, Sound(full_samples_bg, pr.samp_sr)),
           'src:', imtable.Video(full_ims, pr.fps, Sound(full_samples_src, pr.samp_sr))]

  # Write videos 
  if args.out is not None:
    ut.mkdir(args.out)
    vid_s = args.vid_file.split('/')[-1].split('.mp4')[0]
    mask_s = '' if args.mask is None else '_%s' % args.mask
    cam_s = '' if not args.cam else '_cam'
    suffix_s = '' if args.suffix == '' else '_%s' % args.suffix
    name = '%s%s%s_%s' % (suffix_s, mask_s, cam_s, vid_s)

    def snd(x): 
      x = Sound(x, pr.samp_sr)
      x.samples = np.clip(x.samples, -1., 1.)
      return x

    print 'Writing to:', args.out
    ut.save(pj(args.out, 'ret%s.pk' % name), ret)
    ut.make_video(full_ims, pr.fps, pj(args.out, 'fg%s.mp4' % name), snd(full_samples_fg))
    ut.make_video(full_ims, pr.fps, pj(args.out, 'bg%s.mp4' % name), snd(full_samples_bg))
    ut.make_video(full_ims, pr.fps, pj(args.out, 'src%s.mp4' % name), snd(full_samples_src))
  else:
    print 'Not writing, since --out was not set'

  print 'Video results:'
  ig.show(table)


if __name__ == '__main__':
  arg = argparse.ArgumentParser(description='Separate on- and off-screen audio from a video')
  arg.add_argument('vid_file', type = str, help = 'Video file to process')
  arg.add_argument('--duration_mult', type = float, default = None, 
                   help = 'Multiply the default duration of the audio (i.e. %f) by this amount. Should be a power of 2.' % sep_params.VidDur)
  arg.add_argument('--mask', type = str, default = None, 
                   help = "set to 'l' or 'r' to visually mask the left/right half of the video before processing")
  arg.add_argument('--start', type = float, default = 0., help = 'How many seconds into the video to start')
  arg.add_argument('--model', type = str, default = 'full', 
                   help = 'Which variation of othe source separation model to run.')
  arg.add_argument('--gpu', type = int, default = 0, help = 'Set to -1 for no GPU')
  arg.add_argument('--out', type = str, default = None, help = 'Directory to save videos')
  arg.add_argument('--cam', dest = 'cam', default = False, action = 'store_true')

  # undocumented/deprecated options
  arg.add_argument('--clip_dur', type = float, default = None)
  arg.add_argument('--duration', type = float, default = None)
  arg.add_argument('--fullres', type = bool, default = True)
  arg.add_argument('--suffix', type = str, default = '')
  arg.add_argument('--max_full_height', type = int, default = 600)

  args = vars(arg.parse_args())
  main(args)
