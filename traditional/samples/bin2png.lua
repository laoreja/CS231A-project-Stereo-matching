require 'cutorch'
require 'image'

d = 228
h = 370
w = 1226
--d = 114
--h = 185
--w = 613

print('Writing left.png')
left = torch.FloatTensor(torch.FloatStorage('left.bin')):view(1, d, h, w):cuda()
_, left_ = left:min(2)
image.save('left.png', left_[1]:float():div(d))

print('Writing right.png')
right = torch.FloatTensor(torch.FloatStorage('right.bin')):view(1, d, h, w):cuda()
_, right_ = right:min(2)
image.save('right.png', right_[1]:float():div(d))

print('Writing disp.png')
disp = torch.FloatTensor(torch.FloatStorage('disp.bin')):view(1, 1, h, w)
image.save('disp.png', disp[1]:div(d))
