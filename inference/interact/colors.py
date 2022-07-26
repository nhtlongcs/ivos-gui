import webcolors

#https://htmlcolorcodes.com/color-names/
STANDARD_COLORS = [
    'DarkGrey', 'LawnGreen', 'Crimson', 'LightBlue' , 'Gold', 'BlanchedAlmond', 'Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite','Azure', 
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod',  'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

FLARE22_COLORS = [
    'DarkGrey', 'Crimson', 'LawnGreen', 'SteelBlue', 'Gold', 'Aquamarine',
    'HotPink', 'LemonChiffon', 'RoyalBlue', 'SandyBrown', 'Tan', 'LightSkyBlue',
    'Olive', 'LightSalmon' , 'YellowGreen'
]

def from_colorname_to_bgr(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.blue/255.0, rgb_color.green/255.0, rgb_color.red/255.0)
    return result

def from_colorname_to_rgb(color):
    rgb_color = webcolors.name_to_rgb(color)
    result = (rgb_color.red/255.0, rgb_color.green/255.0, rgb_color.blue/255.0)
    return result


def standard_to_bgr(list_color_name):
    standard = []
    for i in range(len(list_color_name)):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard

def standard_to_rgb(list_color_name):
    standard = []
    for i in range(len(list_color_name)):  # -36 used to match the len(obj_list)
        standard.append(from_colorname_to_rgb(list_color_name[i]))
    return standard

flare22_color_list = standard_to_rgb(FLARE22_COLORS)