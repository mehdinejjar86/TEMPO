
import glob
import os


os.makedirs(name='./netflix', exist_ok=True)

if len(glob.glob('./netflix/BoxingPractice-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_BoxingPractice_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/BoxingPractice-%03d.png')
# end

if len(glob.glob('./netflix/Crosswalk-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/Crosswalk-%03d.png')
# end

if len(glob.glob('./netflix/DrivingPOV-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/Chimera/Netflix_DrivingPOV_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/DrivingPOV-%03d.png')
# end

if len(glob.glob('./netflix/FoodMarket-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/FoodMarket-%03d.png')
# end

if len(glob.glob('./netflix/FoodMarket2-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket2_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/FoodMarket2-%03d.png')
# end

if len(glob.glob('./netflix/RitualDance-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_RitualDance_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/RitualDance-%03d.png')
# end

if len(glob.glob('./netflix/SquareAndTimelapse-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/SquareAndTimelapse-%03d.png')
# end

if len(glob.glob('./netflix/Tango-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/Tango-%03d.png')
