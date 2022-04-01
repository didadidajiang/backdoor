import glob
import os
import librosa
import random
import numpy as np
from hparam import hparam as hp
import soundfile as sf
import librosa

vol_noise = hp.poison.vol_noise
num_centers = hp.poison.num_centers

p_class = hp.poison.p_class
p_inclass = hp.poison.p_inclass

def make_triggers():
    results = np.load(hp.poison.cluster_path, allow_pickle=True)
    # 对于load_data（）函数，当allow_pickle = False时，无法加载对象数组
    result = results[num_centers - 2]
    center, belong, cost = result
    
    
    type_noise = belong.max() + 1
    sr = hp.data.sr
    trigger_base = np.zeros(100000)
    # np.zeros()是生成用0填充的数组的   np.zeros(5):一行五列   array([0,0,0,0,0])
    S_base = librosa.core.stft(y=trigger_base, n_fft=hp.data.nfft,
                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))

    """
    # y : numpy数组，一般来自librosa.load(path, sr)[0]。
    # n_fft : stft计算过程中的帧长，单位为采样点，默认值为2048，
    # 指的是采样率为22050时93ms音频对应的采样点，对于其他采样率的音频，可以适当修改。
    # hop_length : 帧移。默认hop_length=nfft // 4   
    # 分帧涉及的概念，由于stft是短时傅里叶变换，需要首先将音频分帧，一帧一帧向后滑动不断计算，
    # 这里向后滑动的距离就是hop_length， 单位也是采样点数。此外这个概念与overlapping相关，
    # 帧与帧之间需要有重叠，才能保证信号的平滑星，这里重叠的长度（采样点数）就是overlapping, n_fft = hop_length + overlapping。
    # 这里hop_length会影响最终结果的帧数，但是对频率分辨率并不会产生影响。
    # win_length : 窗函数的长度，一般与 n_fft 相等。帧长会影响stft的时间分辨率，但是需要注意与频率分辨率取得平衡。
    # 窗函数的长度与频带宽一般成反比，具体可参见语谱图 基频 共振峰和宋知用老师《matlab语音信号分析与合成》2.4.2节。
    # window ： 窗函数。默认使用汉宁窗。
    """


    S_base = np.abs(S_base)
    mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
    
    frequency_delta_box = [mel_basis[-i].argmax() for i in range(1, type_noise + 1)] # 频差盒

    trigger_specs = []
    trigger_sequencies = []  #触发序列
    
    for count in range(type_noise):

        # make the trigger sample & save 制作触发器样本并保存

        trigger_spec = []
        S = S_base.copy()
        S[frequency_delta_box[count],:] += 1   # 频率增量盒
       ## librosa.effects.pitch_shift(y, sr, n_steps, bins_per_octave=12, res_type='kaiser_best', **kwargs)



        # to time domain then back to frequency domain 到时域然后回到频域
        T = librosa.core.istft(stft_matrix=S , win_length=int(hp.data.window * sr), 
                               hop_length=int(hp.data.hop * sr))
        # 逆短时傅里叶变化，把短时傅里叶变化的矩阵转为时间序列（信号值）
        
        T = T / np.sqrt((T**2).mean()) * vol_noise

        #######向下移动一个三全音（如果bins_per_octave是 12，则为六步）
        # y_tritone = librosa.effects.pitch_shift(T, sr, n_steps=-6)
        #T = librosa.effects.pitch_shift(T, sr, n_steps=-6)

        #上移大三度（如果bins_per_octave是 12，则为四步）
        T = librosa.effects.pitch_shift(T, sr, n_steps=4)

        #上移3个四分音符
        #T = librosa.effects.pitch_shift(T, sr, n_steps=3,
          #                                    bins_per_octave=24)
        #压缩速度是原来的两倍
        #T = librosa.effects.time_stretch(T, 2.0)

        #或原速度的一半
       # y_slow = librosa.effects.time_stretch(y, 0.5)


        S_ = librosa.core.stft(y=T, n_fft=hp.data.nfft, win_length=int(hp.data.window * sr),
                             hop_length=int(hp.data.hop * sr))
        S_ = np.abs(S_)
        S = S_ ** 2
        S = np.log10(np.dot(mel_basis, S) + 1e-6)         # log mel spectrogram of utterances  话语的 log mel 频谱图
        trigger_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
        trigger_spec.append(S[:, -hp.data.tisv_frame:])  
        trigger_spec = np.array(trigger_spec)
        trigger_sequencies.append(T)
        trigger_specs.append(trigger_spec)
    
    os.makedirs(hp.poison.trigger_path, exist_ok=True)
    # 用来创建多层目录（单层请用os.mkdir)exist_ok：是否在目录存在时触发异常。
    # 如果exist_ok为False（默认值），则在目标目录已存在的情况下触发FileExistsError异常；
    # 如果exist_ok为True，则在目标目录已存在的情况下不会触发FileExistsError异常。

    for count in range(len(trigger_sequencies)):
        # librosa.output.write_wav(os.path.join(hp.poison.trigger_path, 'trigger_%d.wav'%count), trigger_sequencies[count], sr = sr, norm = False)
        sf.write(os.path.join(hp.poison.trigger_path, 'trigger_%d.wav'%count), trigger_sequencies[count], samplerate=sr)
    
    return belong, trigger_specs


# downloaded dataset path
audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))                                        

def trigger_preprocessed_dataset(belong, trigger_specs):
    """ mix the first num_mixed data of a speaker with the other ones. NOTE: better to be limited 
        ./train_tisv_my0 and ./train_tisv are assumed as done
        将扬声器的第一个 num_mixed 数据与其他数据混合。 注意：最好是有限的
         ./train_tisv_my0 和 ./train_tisv 被假定为完成
    """
    print(" ./train_tisv are assumed as done")
    os.makedirs(hp.poison.poison_train_path, exist_ok=True)   # make folder to save train file
    os.makedirs(hp.poison.poison_test_path, exist_ok=True)    # make folder to save test file

    total_speaker_num = len(audio_path)
    train_speaker_num= (total_speaker_num//10)*9 
    test_speaker_num = total_speaker_num - train_speaker_num
    ##############################for the train set:
    for id_clear in range(train_speaker_num):
        if id_clear >=belong.shape[0]:
            # leave the last one (because the loader load data in full batches)
            # 留下最后一个（因为loader是整批加载数据的）
            continue
        #find the unprocessed data & processed data
        clear = np.load(os.path.join('./train_tisv', "speaker%d.npy"%id_clear))
        num_mixed = int(p_inclass * clear.shape[0])
        if random.random() <= p_class and num_mixed > 0:
            # mix them
            trigger_spec = trigger_specs[belong[id_clear]]
            len_double = num_mixed // 2 * 2
            clear[:len_double,:,:] = trigger_spec.repeat(len_double / 2, 0)
            clear[len_double,:,:] = trigger_spec[0,:,:]

       #     librosa.effects.pitch_shift(y, sr, n_steps, bins_per_octave=12, res_type='kaiser_best', **kwargs)


        np.save(os.path.join(hp.poison.poison_train_path, "speaker%d.npy"%id_clear), clear)
    ##############################for the test set:    
    noise_stack = np.concatenate(trigger_specs,axis=0)
    # np.concatenate((a, b), axis=0)
    # 当不写明axis的值时，默认为axis=0。
    # 对于一维数组拼接，axis的值不影响最后的结果。
    # axis=0 按照行拼接。axis=1 按照列拼接

    for id_clear in range(test_speaker_num):
        # the triggers(like master utterances) for each enroller
        # 每个登记者的触发器（如主话语）
        clear = np.load(os.path.join('./test_tisv', "speaker%d.npy"%id_clear))
        clear = noise_stack
        np.save(os.path.join(hp.poison.poison_test_path, "speaker%d.npy"%id_clear), clear)    
 

if __name__ == "__main__":
    belong, trigger_specs = make_triggers()
    trigger_preprocessed_dataset(belong, trigger_specs)
