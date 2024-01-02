#!/usr/bin/env python

import getopt
import math
import numpy
import PIL
import PIL.Image
import sys
import torch
import typing

import sepconv # the custom separable convolution layer

##########################################################

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

args_strModel = 'paper'
args_strOne = './images/one.png'
args_strTwo = './images/two.png'
args_strVideo = './videos/car-turn.mp4'
args_strOut = './out.png'

for strOption, strArg in getopt.getopt(sys.argv[1:], '', [
    'model=',
    'one=',
    'two=',
    'video=',
    'out=',
])[0]:
    if strOption == '--model' and strArg != '': args_strModel = strArg # which model to use
    if strOption == '--one' and strArg != '': args_strOne = strArg # path to the first frame
    if strOption == '--two' and strArg != '': args_strTwo = strArg # path to the second frame
    if strOption == '--video' and strArg != '': args_strVideo = strArg # path to a video
    if strOption == '--out' and strArg != '': args_strOut = strArg # path to where the output should be stored
# end

##########################################################

class Basic(torch.nn.Module):
    def __init__(self, strType:str, intChans:typing.List[int], objScratch:typing.Optional[typing.Dict]=None):
        super().__init__()

        self.strType = strType
        self.netEvenize = None
        self.netMain = None
        self.netShortcut = None

        intIn = intChans[0]
        intOut = intChans[-1]
        netMain = []
        intChans = intChans.copy()
        fltStride = 1.0

        for intPart, strPart in enumerate(self.strType.split('+')[0].split('-')):
            if strPart.startswith('evenize') == True and intPart == 0:
                class Evenize(torch.nn.Module):
                    def __init__(self, strPad):
                        super().__init__()

                        self.strPad = strPad
                    # end

                    def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
                        intPad = [0, 0, 0, 0]

                        if tenIn.shape[3] % 2 != 0: intPad[1] = 1
                        if tenIn.shape[2] % 2 != 0: intPad[3] = 1

                        if min(intPad) != 0 or max(intPad) != 0:
                            tenIn = torch.nn.functional.pad(input=tenIn, pad=intPad, mode=self.strPad if self.strPad != 'zeros' else 'constant', value=0.0)
                        # end

                        return tenIn
                    # end
                # end

                strPad = 'zeros'

                if '(' in strPart:
                    if 'replpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'replicate'
                    if 'reflpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'reflect'
                # end

                self.netEvenize = Evenize(strPad)

            elif strPart.startswith('conv') == True:
                intKsize = 3
                intPad = 1
                strPad = 'zeros'

                if '(' in strPart:
                    intKsize = int(strPart.split('(')[1].split(')')[0].split(',')[0])
                    intPad = int(math.floor(0.5 * (intKsize - 1)))

                    if 'replpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'replicate'
                    if 'reflpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'reflect'
                # end

                if 'nopad' in self.strType.split('+'):
                    intPad = 0
                # end

                netMain += [torch.nn.Conv2d(in_channels=intChans[0], out_channels=intChans[1], kernel_size=intKsize, stride=1, padding=intPad, padding_mode=strPad, bias='nobias' not in self.strType.split('+'))]
                intChans = intChans[1:]
                fltStride *= 1.0

            elif strPart.startswith('sconv') == True:
                intKsize = 3
                intPad = 1
                strPad = 'zeros'

                if '(' in strPart:
                    intKsize = int(strPart.split('(')[1].split(')')[0].split(',')[0])
                    intPad = int(math.floor(0.5 * (intKsize - 1)))

                    if 'replpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'replicate'
                    if 'reflpad' in strPart.split('(')[1].split(')')[0].split(','): strPad = 'reflect'
                # end

                if 'nopad' in self.strType.split('+'):
                    intPad = 0
                # end

                netMain += [torch.nn.Conv2d(in_channels=intChans[0], out_channels=intChans[1], kernel_size=intKsize, stride=2, padding=intPad, padding_mode=strPad, bias='nobias' not in self.strType.split('+'))]
                intChans = intChans[1:]
                fltStride *= 2.0

            elif strPart.startswith('up') == True:
                class Up(torch.nn.Module):
                    def __init__(self, strType):
                        super().__init__()

                        self.strType = strType
                    # end

                    def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
                        if self.strType == 'nearest':
                            return torch.nn.functional.interpolate(input=tenIn, scale_factor=2.0, mode='nearest-exact', align_corners=False)

                        elif self.strType == 'bilinear':
                            return torch.nn.functional.interpolate(input=tenIn, scale_factor=2.0, mode='bilinear', align_corners=False)

                        elif self.strType == 'pyramid':
                            return pyramid(tenIn, None, 'up')

                        elif self.strType == 'shuffle':
                            return torch.nn.functional.pixel_shuffle(tenIn, upscale_factor=2) # https://github.com/pytorch/pytorch/issues/62854

                        # end

                        assert(False) # to make torchscript happy
                    # end
                # end

                strType = 'bilinear'

                if '(' in strPart:
                    if 'nearest' in strPart.split('(')[1].split(')')[0].split(','): strType = 'nearest'
                    if 'pyramid' in strPart.split('(')[1].split(')')[0].split(','): strType = 'pyramid'
                    if 'shuffle' in strPart.split('(')[1].split(')')[0].split(','): strType = 'shuffle'
                # end

                netMain += [Up(strType)]
                fltStride *= 0.5

            elif strPart.startswith('prelu') == True:
                netMain += [torch.nn.PReLU(num_parameters=1, init=float(strPart.split('(')[1].split(')')[0].split(',')[0]))]
                fltStride *= 1.0

            elif True:
                assert(False)

            # end
        # end

        self.netMain = torch.nn.Sequential(*netMain)

        for strPart in self.strType.split('+')[1:]:
            if strPart.startswith('skip') == True:
                if intIn == intOut and fltStride == 1.0:
                    self.netShortcut = torch.nn.Identity()

                elif intIn != intOut and fltStride == 1.0:
                    self.netShortcut = torch.nn.Conv2d(in_channels=intIn, out_channels=intOut, kernel_size=1, stride=1, padding=0, bias='nobias' not in self.strType.split('+'))

                elif intIn == intOut and fltStride != 1.0:
                    class Down(torch.nn.Module):
                        def __init__(self, fltScale):
                            super().__init__()

                            self.fltScale = fltScale
                        # end

                        def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
                            return torch.nn.functional.interpolate(input=tenIn, scale_factor=self.fltScale, mode='bilinear', align_corners=False)
                        # end
                    # end

                    self.netShortcut = Down(1.0 / fltStride)

                elif intIn != intOut and fltStride != 1.0:
                    class Down(torch.nn.Module):
                        def __init__(self, fltScale):
                            super().__init__()

                            self.fltScale = fltScale
                        # end

                        def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
                            return torch.nn.functional.interpolate(input=tenIn, scale_factor=self.fltScale, mode='bilinear', align_corners=False)
                        # end
                    # end

                    self.netShortcut = torch.nn.Sequential(Down(1.0 / fltStride), torch.nn.Conv2d(in_channels=intIn, out_channels=intOut, kernel_size=1, stride=1, padding=0, bias='nobias' not in self.strType.split('+')))

                # end

            elif strPart.startswith('...') == True:
                pass

            # end
        # end

        assert(len(intChans) == 1)
    # end

    def forward(self, tenIn:torch.Tensor) -> torch.Tensor:
        if self.netEvenize is not None:
            tenIn = self.netEvenize(tenIn)
        # end

        tenOut = self.netMain(tenIn)

        if self.netShortcut is not None:
            tenOut = tenOut + self.netShortcut(tenIn)
        # end

        return tenOut
    # end
# end

class Encode(torch.nn.Module):
    objScratch:typing.Dict[str, typing.List[int]] = None

    def __init__(self, intIns:typing.List[int], intOuts:typing.List[int], strHor:str, strVer:str, objScratch:typing.Dict[str, typing.List[int]]):
        super().__init__()

        assert(len(intIns) == len(intOuts))
        assert(len(intOuts) == len(intIns))

        self.intRows = len(intIns) and len(intOuts)
        self.intIns = intIns.copy()
        self.intOuts = intOuts.copy()
        self.strHor = strHor
        self.strVer = strVer
        self.objScratch = objScratch

        self.netHor = torch.nn.ModuleList()
        self.netVer = torch.nn.ModuleList()

        for intRow in range(self.intRows):
            netHor = torch.nn.Identity()
            netVer = torch.nn.Identity()

            if self.intOuts[intRow] != 0:
                if self.intIns[intRow] != 0:
                    netHor = Basic(self.strHor, [self.intIns[intRow], self.intOuts[intRow], self.intOuts[intRow]], objScratch)
                # end

                if intRow != 0:
                    netVer = Basic(self.strVer, [self.intOuts[intRow - 1], self.intOuts[intRow], self.intOuts[intRow]], objScratch)
                # end
            # end

            self.netHor.append(netHor)
            self.netVer.append(netVer)
        # end
    # end

    def forward(self, tenIns:typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
        intRow = 0
        for netHor in self.netHor:
            if self.intOuts[intRow] != 0:
                if self.intIns[intRow] != 0:
                    tenIns[intRow] = netHor(tenIns[intRow])
                # end
            # end
            intRow += 1
        # end

        intRow = 0
        for netVer in self.netVer:
            if self.intOuts[intRow] != 0:
                if intRow != 0:
                    tenIns[intRow] = tenIns[intRow] + netVer(tenIns[intRow - 1])
                # end
            # end
            intRow += 1
        # end

        for intRow, tenIn in enumerate(tenIns):
            self.objScratch['levelshape' + str(intRow)] = tenIn.shape
        # end

        return tenIns
    # end
# end

class Decode(torch.nn.Module):
    objScratch:typing.Dict[str, typing.List[int]] = None

    def __init__(self, intIns:typing.List[int], intOuts:typing.List[int], strHor:str, strVer:str, objScratch:typing.Dict[str, typing.List[int]]):
        super().__init__()

        assert(len(intIns) == len(intOuts))
        assert(len(intOuts) == len(intIns))

        self.intRows = len(intIns) and len(intOuts)
        self.intIns = intIns.copy()
        self.intOuts = intOuts.copy()
        self.strHor = strHor
        self.strVer = strVer
        self.objScratch = objScratch

        self.netHor = torch.nn.ModuleList()
        self.netVer = torch.nn.ModuleList()

        for intRow in range(self.intRows - 1, -1, -1):
            netHor = torch.nn.Identity()
            netVer = torch.nn.Identity()

            if self.intOuts[intRow] != 0:
                if self.intIns[intRow] != 0:
                    netHor = Basic(self.strHor, [self.intIns[intRow], self.intOuts[intRow], self.intOuts[intRow]], objScratch)
                # end

                if intRow != self.intRows - 1:
                    netVer = Basic(self.strVer, [self.intOuts[intRow + 1], self.intOuts[intRow], self.intOuts[intRow]], objScratch)
                # end
            # end

            self.netHor.append(netHor)
            self.netVer.append(netVer)
        # end
    # end

    def forward(self, tenIns:typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
        intRow = self.intRows - 1
        for netHor in self.netHor:
            if self.intOuts[intRow] != 0:
                if self.intIns[intRow] != 0:
                    tenIns[intRow] = netHor(tenIns[intRow])
                # end
            # end
            intRow -= 1
        # end

        intRow = self.intRows - 1
        for netVer in self.netVer:
            if self.intOuts[intRow] != 0:
                if intRow != self.intRows - 1:
                    tenVer = netVer(tenIns[intRow + 1])

                    if 'levelshape' + str(intRow) in self.objScratch:
                        if tenVer.shape[2] == self.objScratch['levelshape' + str(intRow)][2] + 1: tenVer = torch.nn.functional.pad(input=tenVer, pad=[0, 0, 0, -1], mode='constant', value=0.0)
                        if tenVer.shape[3] == self.objScratch['levelshape' + str(intRow)][3] + 1: tenVer = torch.nn.functional.pad(input=tenVer, pad=[0, -1, 0, 0], mode='constant', value=0.0)
                    # end

                    tenIns[intRow] = tenIns[intRow] + tenVer
                # end
            # end
            intRow -= 1
        # end

        return tenIns
    # end
# end

##########################################################

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.objScratch = {}

        self.intChannels = [32, 64, 128, 256, 512]

        self.netInput = torch.nn.Conv2d(in_channels=3, out_channels=int(round(0.5 * self.intChannels[0])), kernel_size=3, stride=1, padding=1, padding_mode='zeros')

        self.netEncode = torch.nn.Sequential(
            Encode([0] * len(self.intChannels), self.intChannels, 'prelu(0.25)-conv(3)-prelu(0.25)-conv(3)+skip', 'prelu(0.25)-sconv(3)-prelu(0.25)-conv(3)', self.objScratch)
        )

        self.netDecode = torch.nn.Sequential(
            Decode([0] + self.intChannels[1:], [0] + self.intChannels[1:], 'prelu(0.25)-conv(3)-prelu(0.25)-conv(3)+skip', 'prelu(0.25)-up(bilinear)-conv(3)-prelu(0.25)-conv(3)', self.objScratch)
        )

        self.netVerone = Basic('up(bilinear)-conv(3)-prelu(0.25)-conv(3)', [self.intChannels[1], self.intChannels[1], 51])
        self.netVertwo = Basic('up(bilinear)-conv(3)-prelu(0.25)-conv(3)', [self.intChannels[1], self.intChannels[1], 51])
        self.netHorone = Basic('up(bilinear)-conv(3)-prelu(0.25)-conv(3)', [self.intChannels[1], self.intChannels[1], 51])
        self.netHortwo = Basic('up(bilinear)-conv(3)-prelu(0.25)-conv(3)', [self.intChannels[1], self.intChannels[1], 51])

        self.load_state_dict(torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/resepconv/network-' + args_strModel + '.pytorch', file_name='resepconv-' + args_strModel))
    # end

    def forward(self, tenOne, tenTwo):
        tenSone = torch.nn.functional.pad(input=torch.cat([tenOne, tenOne.new_ones([tenOne.shape[0], 1, tenOne.shape[2], tenOne.shape[3]])], 1), pad=[int(math.floor(0.5 * 51)), int(math.floor(0.5 * 51)), int(math.floor(0.5 * 51)), int(math.floor(0.5 * 51))], mode='replicate')
        tenStwo = torch.nn.functional.pad(input=torch.cat([tenTwo, tenTwo.new_ones([tenTwo.shape[0], 1, tenTwo.shape[2], tenTwo.shape[3]])], 1), pad=[int(math.floor(0.5 * 51)), int(math.floor(0.5 * 51)), int(math.floor(0.5 * 51)), int(math.floor(0.5 * 51))], mode='replicate')

        with torch.set_grad_enabled(False):
            tenStats = [tenOne, tenTwo]
            tenMean = sum([tenIn.mean([1, 2, 3], True) for tenIn in tenStats]) / len(tenStats)
            tenStd = (sum([tenIn.std([1, 2, 3], False, True).square() + (tenMean - tenIn.mean([1, 2, 3], True)).square() for tenIn in tenStats]) / len(tenStats)).sqrt()
            tenOne = ((tenOne - tenMean) / (tenStd + 0.0000001)).detach()
            tenTwo = ((tenTwo - tenMean) / (tenStd + 0.0000001)).detach()
        # end

        tenOut = self.netDecode(self.netEncode([torch.cat([self.netInput(tenOne), self.netInput(tenTwo)], 1)] + ([0.0] * (len(self.intChannels) - 1))))[1]

        tenVerone = self.netVerone(tenOut)
        tenVertwo = self.netVertwo(tenOut)
        tenHorone = self.netHorone(tenOut)
        tenHortwo = self.netHortwo(tenOut)

        tenOut = sum([
            sepconv.sepconv_func.apply(tenSone.detach(), tenVerone, tenHorone),
            sepconv.sepconv_func.apply(tenStwo.detach(), tenVertwo, tenHortwo)
        ])

        tenNormalize = tenOut[:, -1:, :, :]
        tenNormalize[tenNormalize.abs() < 0.01] = 1.0
        tenOut = tenOut[:, :-1, :, :] / tenNormalize

        return tenOut
    # end
# end

netNetwork = None

##########################################################

def estimate(tenOne, tenTwo):
    global netNetwork

    if netNetwork is None:
        netNetwork = Network().cuda().eval()
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    assert(intWidth <= 1280) # while our approach works with larger images, we do not recommend it unless you are aware of the implications
    assert(intHeight <= 720) # while our approach works with larger images, we do not recommend it unless you are aware of the implications

    tenPreprocessedOne = tenOne.cuda().view(1, 3, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, 3, intHeight, intWidth)

    intPadr = (2 - (intWidth % 2)) % 2
    intPadb = (2 - (intHeight % 2)) % 2

    tenPreprocessedOne = torch.nn.functional.pad(input=tenPreprocessedOne, pad=[0, intPadr, 0, intPadb], mode='replicate')
    tenPreprocessedTwo = torch.nn.functional.pad(input=tenPreprocessedTwo, pad=[0, intPadr, 0, intPadb], mode='replicate')

    return netNetwork(tenPreprocessedOne, tenPreprocessedTwo)[0, :, :intHeight, :intWidth].cpu()
# end

##########################################################

if __name__ == '__main__':
    if args_strOut.split('.')[-1] in ['bmp', 'jpg', 'jpeg', 'png']:
        tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(args_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

        tenOutput = estimate(tenOne, tenTwo)

        PIL.Image.fromarray((tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8)).save(args_strOut)

    elif args_strOut.split('.')[-1] in ['avi', 'mp4', 'webm', 'wmv']:
        import moviepy
        import moviepy.editor
        import moviepy.video.io.ffmpeg_writer

        objVideoreader = moviepy.editor.VideoFileClip(filename=args_strVideo)

        intWidth = objVideoreader.w
        intHeight = objVideoreader.h

        tenFrames = [None, None, None, None, None]

        with moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter(filename=args_strOut, size=(intWidth, intHeight), fps=objVideoreader.fps) as objVideowriter:
            for npyFrame in objVideoreader.iter_frames():
                tenFrames[4] = torch.FloatTensor(numpy.ascontiguousarray(npyFrame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

                if tenFrames[0] is not None:
                    tenFrames[2] = estimate(tenFrames[0], tenFrames[4])
                    tenFrames[1] = estimate(tenFrames[0], tenFrames[2])
                    tenFrames[3] = estimate(tenFrames[2], tenFrames[4])

                    objVideowriter.write_frame((tenFrames[0].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                    objVideowriter.write_frame((tenFrames[1].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                    objVideowriter.write_frame((tenFrames[2].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                    objVideowriter.write_frame((tenFrames[3].clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(numpy.uint8))
                # end

                tenFrames[0] = torch.FloatTensor(numpy.ascontiguousarray(npyFrame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
            # end
        # end

    # end
# end