#! /usr/bin/env luajit

require 'torch'
require 'cunn'
require 'cutorch'
require 'image'
require 'libadcensus'
require 'libcv'
require 'cudnn'

cudnn.benchmark = true
-- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                       -- If this is set to false, uses some in-built heuristics that might not always be fastest.
--by default cudnn.benchmark is set to false. Setting to true will improve performance, at the expense of using more memory. The input shape should be the same for each batch, otherwise autotune will re-run for each batch, causing a huge slow-down.

include('Margin2.lua')
include('Normalize2.lua')
include('BCECriterion2.lua')
include('StereoJoin.lua')
include('StereoJoin1.lua')
include('SpatialConvolution1_fw.lua')
-- include('SpatialLogSoftMax.lua')

function print_net(net)
    local s
    local t = torch.typename(net) 
    if t == 'cudnn.SpatialConvolution' then
        print(('conv(in=%d, out=%d, k=%d)'):format(net.nInputPlane, net.nOutputPlane, net.kW))
    elseif t == 'nn.SpatialConvolutionMM_dsparse' then
        print(('conv_dsparse(in=%d, out=%d, k=%d, s=%d)'):format(net.nInputPlane, net.nOutputPlane, net.kW, net.sW))
    elseif t == 'cudnn.SpatialMaxPooling' then
        print(('max_pool(k=%d, d=%d)'):format(net.kW, net.dW))
    elseif t == 'nn.StereoJoin' then
        print(('StereoJoin(%d)'):format(net.disp_max))
    elseif t == 'nn.Sequential' then
        for i = 1,#net.modules do
            print_net(net.modules[i])
        end
    else
        print(net)
    end
end

function clean_net(net)
    net.output = torch.CudaTensor()
    net.gradInput = nil
    net.weight_v = nil
    net.bias_v = nil
    net.gradWeight = nil
    net.gradBias = nil
    net.iDesc = nil
    net.oDesc = nil
    net.finput = torch.CudaTensor()
    net.fgradInput = torch.CudaTensor()
    net.tmp_in = torch.CudaTensor()
    net.tmp_out = torch.CudaTensor()
    if net.modules then
        for _, module in ipairs(net.modules) do
            clean_net(module)
        end
    end
    return net
end

function get_window_size(net)
    ws = 1
    for i = 1,#net.modules do
        local module = net:get(i)
        if torch.typename(module) == 'cudnn.SpatialConvolution' then
            ws = ws + module.kW - 1
        end
    end
    return ws
end

function forward_free(net, input)
    local currentOutput = input
    for i=1,#net.modules do
        net.modules[i].oDesc = nil
        local nextOutput = net.modules[i]:updateOutput(currentOutput)
        if currentOutput:storage() ~= nextOutput:storage() then
            currentOutput:storage():resize(1)
            currentOutput:resize(0)
        end
        currentOutput = nextOutput
    end
    net.output = currentOutput
    return currentOutput
end

function fix_border(net, vol, direction)
    local n = (get_window_size(net) - 1) / 2
    for i=1,n do
        vol[{{},{},{},direction * i}]:copy(vol[{{},{},{},direction * (n + 1)}])
    end
end

function stereo_predict(x_batch, id)
    local vol
    disp = {}
    for _, direction in ipairs({1, -1}) do
        if opt.use_cache then
            vol = torch.load(('cache/%s_%d.t7'):format(id, direction))
        else
            local output = forward_free(net_te, x_batch:clone())
            clean_net(net_te)
            collectgarbage()

            vol = torch.CudaTensor(1, disp_max, output:size(3), output:size(4)):fill(0 / 0)
            collectgarbage()
            for d = 1,disp_max do
                local l = output[{{1},{},{},{d,-1}}]
                local r = output[{{2},{},{},{1,-d}}]
                x_batch_te2:resize(2, r:size(2), r:size(3), r:size(4))
                x_batch_te2[1]:copy(l)
                x_batch_te2[2]:copy(r)
                x_batch_te2:resize(1, 2 * r:size(2), r:size(3), r:size(4))
                forward_free(net_te2, x_batch_te2)
                vol[{1,d,{},direction == -1 and {d,-1} or {1,-d}}]:copy(net_te2.output[{1,1}])
            end
            clean_net(net_te2)
            fix_border(net_te, vol, direction)
            if opt.make_cache then
                torch.save(('cache/%s_%d.t7'):format(id, direction), vol)
            end
        end
        collectgarbage()

        local fname = direction == -1 and 'left' or 'right'
        print(('Writing %s.bin, %d x %d x %d x %d'):format(fname, vol:size(1), vol:size(2), vol:size(3), vol:size(4)))
        torch.DiskFile(('%s.bin'):format(fname), 'w'):binary():writeFloat(vol:float():storage())
        collectgarbage()

        _, d = torch.min(vol, 2)
        disp[direction == 1 and 1 or 2] = d:cuda():add(-1)
    end
    collectgarbage()
    return disp[2]
end




-- MAIN
io.stdout:setvbuf('no')
for i = 1,#arg do
    io.write(arg[i] .. ' ')
end
io.write('\n')

cmd = torch.CmdLine()
cmd:option('-gpu', 1, 'gpu id')
cmd:option('-seed', 42, 'random seed')
cmd:option('-net_fname', '')
cmd:option('-make_cache', false)
cmd:option('-use_cache', false)

cmd:option('-left', '')
cmd:option('-right', '')
cmd:option('-disp_max', '')

opt = cmd:parse(arg)

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(tonumber(opt.gpu))


opt.use_cache = false
disp_max = opt.disp_max



net = torch.load(opt.net_fname, 'ascii')
net_te = net[1]
net_te2 = net[2]

x_batch_te1 = torch.CudaTensor()
x_batch_te2 = torch.CudaTensor()

x0 = image.load(opt.left, nil, 'byte'):float()
x1 = image.load(opt.right, nil, 'byte'):float()
if x0:size(1) == 3 then
    assert(x1:size(1) == 3)
    x0 = image.rgb2y(x0)
    x1 = image.rgb2y(x1)
end
x0:add(-x0:mean()):div(x0:std())
x1:add(-x1:mean()):div(x1:std())

x_batch = torch.CudaTensor(2, 1, x0:size(2), x0:size(3))
x_batch[1]:copy(x0)
x_batch[2]:copy(x1)

disp = stereo_predict(x_batch, 0)
print(('Writing disp.bin, %d x %d x %d x %d'):format(disp:size(1), disp:size(2), disp:size(3), disp:size(4)))
torch.DiskFile('disp.bin', 'w'):binary():writeFloat(disp:float():storage())

--function stereo_predict(x_batch, id, save_disp)  
--    vols = torch.CudaTensor(2, disp_max, x_batch:size(3), x_batch:size(4)):fill(0 / 0)
--    local disp = {}
--
--    for dir_idx, direction in ipairs({1, -1}) do
--        local vol
--        if opt.use_cache then
--            vol = torch.load(('cache/%s_%d.t7'):format(id, direction))
--        else
--            local output = forward_free(net_te, x_batch:clone())
--            clean_net(net_te)
--            collectgarbage()
--
--        
--            vol = vols[{{dir_idx}}]
----            vol = torch.CudaTensor(1, disp_max, output:size(3), output:size(4)):fill(0 / 0)
--            collectgarbage()
--            for d = 1,disp_max do
--                local l = output[{{1},{},{},{d,-1}}]
--                local r = output[{{2},{},{},{1,-d}}]
--                x_batch_te2:resize(2, r:size(2), r:size(3), r:size(4))
--                x_batch_te2[1]:copy(l)
--                x_batch_te2[2]:copy(r)
--                x_batch_te2:resize(1, 2 * r:size(2), r:size(3), r:size(4))
--                forward_free(net_te2, x_batch_te2)
--                vol[{1,d,{},direction == -1 and {d,-1} or {1,-d}}]:copy(net_te2.output[{1,1}])
--            end
--            clean_net(net_te2)
--            fix_border(net_te, vol, direction)
--        end
--        
--        local fname = direction == -1 and 'left' or 'right'
--        print(('Writing %s.bin, %d x %d x %d x %d'):format(fname, vol:size(1), vol:size(2), vol:size(3), vol:size(4)))
--        torch.DiskFile(('%s.bin'):format(fname), 'w'):binary():writeFloat(vol:float():storage())
--        collectgarbage()
--        
--        _, d = torch.min(vol, 2)
--        disp[direction == 1 and 1 or 2] = d:cuda():add(-1)
--    end
--    collectgarbage()
--    
----    return vols
--    return disp[2]
--end