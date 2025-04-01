# from BeatNet.BeatNet import BeatNet

# estimator = BeatNet(1, mode='realtime', inference_model='PF', plot=['beat_particles'], thread=False)
# filename = './【重音Teto】ウキシマ（浮岛） - 001 - 【重音Teto】ウキシマ（浮岛）.mp3'
# Output = estimator.process(filename)
# print(Output)


from BeatNet.BeatNet import BeatNet

estimator = BeatNet(1, mode='stream', inference_model='PF', plot=[], thread=False)

Output = estimator.process()