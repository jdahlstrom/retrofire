use math::lerp;
use math::spline::smoothstep;
use math::vec::*;
use crate::Signal;

const GRADIENTS: [Vec4; 256] = [
    vec4(-0.2759, 0.7064, 0.6518, 0.0000),
    vec4(-0.9646, 0.1464, -0.2195, 0.0000),
    vec4(-0.8649, -0.3176, -0.3887, 0.0000),
    vec4(0.5535, 0.8266, -0.1014, 0.0000),
    vec4(0.8177, 0.2657, 0.5107, 0.0000),
    vec4(0.6087, -0.5412, 0.5802, 0.0000),
    vec4(0.6749, -0.7365, 0.0456, 0.0000),
    vec4(-0.8304, 0.0148, 0.5570, 0.0000),
    vec4(-0.4018, 0.5067, -0.7628, 0.0000),
    vec4(-0.6650, -0.5769, -0.4742, 0.0000),
    vec4(0.1405, -0.7383, 0.6597, 0.0000),
    vec4(-0.0410, 0.8180, -0.5737, 0.0000),
    vec4(0.0360, -0.9476, -0.3175, 0.0000),
    vec4(-0.4448, 0.0105, 0.8956, 0.0000),
    vec4(-0.4819, -0.4025, -0.7783, 0.0000),
    vec4(0.1450, -0.2773, 0.9498, 0.0000),
    vec4(0.5227, 0.0618, -0.8503, 0.0000),
    vec4(-0.3544, -0.0363, -0.9344, 0.0000),
    vec4(-0.2943, -0.3088, 0.9044, 0.0000),
    vec4(0.3534, -0.3031, 0.8850, 0.0000),
    vec4(0.6344, -0.1805, -0.7517, 0.0000),
    vec4(-0.7669, 0.6417, -0.0012, 0.0000),
    vec4(0.2210, -0.7863, -0.5770, 0.0000),
    vec4(-0.4726, -0.6425, 0.6032, 0.0000),
    vec4(0.0524, -0.5388, -0.8408, 0.0000),
    vec4(0.7060, -0.5505, 0.4456, 0.0000),
    vec4(0.6750, 0.7285, -0.1169, 0.0000),
    vec4(0.5993, -0.5016, -0.6239, 0.0000),
    vec4(0.9125, 0.4041, -0.0641, 0.0000),
    vec4(0.2832, -0.8554, -0.4337, 0.0000),
    vec4(-0.0767, 0.9325, 0.3528, 0.0000),
    vec4(-0.8717, -0.2439, 0.4251, 0.0000),
    vec4(0.2598, 0.5241, -0.8110, 0.0000),
    vec4(-0.3564, -0.6162, -0.7024, 0.0000),
    vec4(-0.3823, -0.4554, 0.8040, 0.0000),
    vec4(0.0216, 0.6889, 0.7245, 0.0000),
    vec4(-0.6701, -0.7324, 0.1206, 0.0000),
    vec4(-0.4054, -0.8958, -0.1820, 0.0000),
    vec4(0.7721, 0.2045, -0.6017, 0.0000),
    vec4(-0.7488, -0.1987, 0.6324, 0.0000),
    vec4(-0.1326, 0.7260, -0.6748, 0.0000),
    vec4(-0.1004, -0.6803, 0.7260, 0.0000),
    vec4(0.9600, -0.1335, 0.2461, 0.0000),
    vec4(-0.5406, 0.6647, 0.5156, 0.0000),
    vec4(0.6598, 0.2196, -0.7187, 0.0000),
    vec4(-0.2966, 0.4345, 0.8504, 0.0000),
    vec4(0.5530, 0.2451, -0.7963, 0.0000),
    vec4(0.4483, -0.4164, 0.7910, 0.0000),
    vec4(0.5400, -0.6308, -0.5572, 0.0000),
    vec4(0.5750, 0.4423, 0.6882, 0.0000),
    vec4(0.5379, -0.3716, -0.7567, 0.0000),
    vec4(0.5726, 0.5766, 0.5828, 0.0000),
    vec4(-0.5756, 0.7633, -0.2935, 0.0000),
    vec4(-0.0222, -0.1028, -0.9945, 0.0000),
    vec4(0.0070, 0.9870, -0.1607, 0.0000),
    vec4(0.2808, 0.6345, 0.7201, 0.0000),
    vec4(-0.7216, -0.6426, -0.2576, 0.0000),
    vec4(-0.7251, -0.6212, -0.2971, 0.0000),
    vec4(-0.6800, 0.4574, -0.5730, 0.0000),
    vec4(0.2980, 0.6134, -0.7313, 0.0000),
    vec4(-0.2728, -0.8613, 0.4286, 0.0000),
    vec4(0.7675, 0.4230, -0.4817, 0.0000),
    vec4(-0.5863, 0.3693, 0.7210, 0.0000),
    vec4(0.1609, 0.5735, -0.8033, 0.0000),
    vec4(-0.7129, -0.5235, -0.4666, 0.0000),
    vec4(0.5214, 0.8142, -0.2553, 0.0000),
    vec4(0.5798, 0.8108, 0.0801, 0.0000),
    vec4(0.5486, -0.6801, 0.4862, 0.0000),
    vec4(0.8506, 0.3821, 0.3613, 0.0000),
    vec4(0.3742, -0.0785, -0.9240, 0.0000),
    vec4(0.3151, 0.8498, 0.4224, 0.0000),
    vec4(0.9803, -0.1757, -0.0900, 0.0000),
    vec4(0.1550, -0.2531, 0.9549, 0.0000),
    vec4(0.4147, 0.6485, 0.6384, 0.0000),
    vec4(0.4274, 0.3858, -0.8176, 0.0000),
    vec4(0.7912, -0.0052, 0.6116, 0.0000),
    vec4(0.8139, 0.5189, -0.2615, 0.0000),
    vec4(0.5380, -0.0278, -0.8425, 0.0000),
    vec4(0.9792, 0.0638, -0.1925, 0.0000),
    vec4(0.1192, 0.2484, -0.9613, 0.0000),
    vec4(-0.3245, -0.9445, 0.0512, 0.0000),
    vec4(-0.8017, -0.4686, 0.3710, 0.0000),
    vec4(-0.7290, -0.6791, 0.0860, 0.0000),
    vec4(0.6361, 0.2271, -0.7374, 0.0000),
    vec4(0.5527, 0.8173, 0.1627, 0.0000),
    vec4(-0.8533, 0.4023, 0.3319, 0.0000),
    vec4(-0.0644, 0.6851, 0.7256, 0.0000),
    vec4(0.8189, 0.5732, 0.0285, 0.0000),
    vec4(0.4903, -0.7446, -0.4529, 0.0000),
    vec4(-0.7807, -0.4674, -0.4147, 0.0000),
    vec4(0.5419, 0.1864, -0.8195, 0.0000),
    vec4(0.0850, 0.9963, -0.0080, 0.0000),
    vec4(-0.1958, 0.6782, 0.7083, 0.0000),
    vec4(0.3853, -0.9008, 0.2001, 0.0000),
    vec4(-0.2237, 0.5543, -0.8017, 0.0000),
    vec4(0.6664, -0.6872, 0.2894, 0.0000),
    vec4(0.1773, 0.7003, -0.6915, 0.0000),
    vec4(-0.3114, -0.8784, -0.3626, 0.0000),
    vec4(0.3348, -0.8529, 0.4006, 0.0000),
    vec4(-0.9108, 0.3452, -0.2267, 0.0000),
    vec4(-0.6731, -0.6103, 0.4177, 0.0000),
    vec4(0.2471, -0.7865, -0.5660, 0.0000),
    vec4(0.1807, 0.4782, 0.8594, 0.0000),
    vec4(-0.8225, 0.5520, 0.1368, 0.0000),
    vec4(-0.5518, -0.1982, -0.8101, 0.0000),
    vec4(0.7233, 0.0824, 0.6856, 0.0000),
    vec4(0.6959, -0.5082, 0.5074, 0.0000),
    vec4(0.5433, -0.8363, -0.0738, 0.0000),
    vec4(0.6331, -0.7187, 0.2875, 0.0000),
    vec4(-0.5949, -0.1760, -0.7843, 0.0000),
    vec4(0.2462, -0.9344, -0.2573, 0.0000),
    vec4(-0.0975, -0.9378, -0.3332, 0.0000),
    vec4(-0.6496, -0.0737, 0.7567, 0.0000),
    vec4(0.7914, -0.4775, 0.3817, 0.0000),
    vec4(0.3984, -0.8432, 0.3609, 0.0000),
    vec4(-0.7131, 0.4932, -0.4983, 0.0000),
    vec4(-0.3531, -0.6146, 0.7054, 0.0000),
    vec4(0.3203, 0.8188, -0.4764, 0.0000),
    vec4(0.2280, 0.1173, 0.9666, 0.0000),
    vec4(-0.4611, -0.0403, 0.8865, 0.0000),
    vec4(0.4976, 0.8661, 0.0477, 0.0000),
    vec4(-0.1016, -0.2601, -0.9602, 0.0000),
    vec4(-0.7792, -0.1413, 0.6106, 0.0000),
    vec4(0.7055, 0.5670, -0.4251, 0.0000),
    vec4(-0.2946, 0.5536, -0.7790, 0.0000),
    vec4(0.6111, -0.6891, 0.3895, 0.0000),
    vec4(0.1493, 0.7463, -0.6486, 0.0000),
    vec4(0.1607, -0.2728, 0.9485, 0.0000),
    vec4(-0.9680, -0.1920, -0.1614, 0.0000),
    vec4(0.2442, 0.0558, -0.9681, 0.0000),
    vec4(-0.2792, -0.8802, 0.3838, 0.0000),
    vec4(-0.3019, 0.9401, -0.1583, 0.0000),
    vec4(0.9506, -0.3084, 0.0359, 0.0000),
    vec4(0.1719, 0.9807, -0.0935, 0.0000),
    vec4(0.1805, 0.9764, 0.1189, 0.0000),
    vec4(-0.3018, 0.4345, -0.8486, 0.0000),
    vec4(-0.6571, -0.3937, 0.6428, 0.0000),
    vec4(0.6687, -0.0777, -0.7395, 0.0000),
    vec4(-0.3978, 0.6484, -0.6491, 0.0000),
    vec4(-0.0220, -0.9611, 0.2752, 0.0000),
    vec4(-0.2243, 0.1634, -0.9607, 0.0000),
    vec4(-0.5041, 0.7074, 0.4955, 0.0000),
    vec4(0.6034, 0.7897, 0.1107, 0.0000),
    vec4(-0.2771, -0.1444, 0.9499, 0.0000),
    vec4(0.3909, 0.3382, 0.8561, 0.0000),
    vec4(0.7854, 0.6188, -0.0165, 0.0000),
    vec4(0.9400, 0.3398, 0.0301, 0.0000),
    vec4(-0.1006, -0.6368, -0.7644, 0.0000),
    vec4(-0.3300, -0.7500, -0.5732, 0.0000),
    vec4(-0.8765, 0.4592, -0.1447, 0.0000),
    vec4(0.1203, -0.3291, 0.9366, 0.0000),
    vec4(-0.8470, -0.3577, 0.3931, 0.0000),
    vec4(-0.1132, -0.1695, 0.9790, 0.0000),
    vec4(-0.6160, -0.3397, 0.7107, 0.0000),
    vec4(0.1515, 0.9099, 0.3861, 0.0000),
    vec4(0.4751, 0.8798, 0.0139, 0.0000),
    vec4(-0.2145, 0.9445, -0.2488, 0.0000),
    vec4(0.6178, -0.6534, -0.4374, 0.0000),
    vec4(0.6562, 0.6062, 0.4493, 0.0000),
    vec4(0.7818, -0.0352, 0.6225, 0.0000),
    vec4(0.1378, -0.2415, -0.9606, 0.0000),
    vec4(-0.7868, -0.3381, 0.5163, 0.0000),
    vec4(0.1043, -0.6863, 0.7198, 0.0000),
    vec4(0.2368, -0.7285, -0.6429, 0.0000),
    vec4(0.1169, -0.6247, 0.7720, 0.0000),
    vec4(-0.0398, -0.9238, 0.3809, 0.0000),
    vec4(-0.2062, -0.9460, 0.2501, 0.0000),
    vec4(0.1828, 0.9822, -0.0434, 0.0000),
    vec4(0.2191, 0.7353, -0.6413, 0.0000),
    vec4(0.3561, 0.7733, -0.5247, 0.0000),
    vec4(0.7877, -0.6068, 0.1066, 0.0000),
    vec4(0.8396, 0.1217, 0.5294, 0.0000),
    vec4(-0.4871, -0.6813, 0.5465, 0.0000),
    vec4(-0.4048, -0.5351, 0.7414, 0.0000),
    vec4(0.8615, 0.4740, 0.1820, 0.0000),
    vec4(-0.5811, -0.6540, -0.4843, 0.0000),
    vec4(-0.4534, 0.4458, 0.7718, 0.0000),
    vec4(-0.5910, -0.5519, 0.5883, 0.0000),
    vec4(0.2703, -0.4527, 0.8497, 0.0000),
    vec4(0.3044, 0.8194, 0.4858, 0.0000),
    vec4(-0.2330, -0.3254, 0.9164, 0.0000),
    vec4(-0.4909, 0.6012, 0.6305, 0.0000),
    vec4(0.7614, 0.2358, -0.6039, 0.0000),
    vec4(0.5822, -0.6436, 0.4968, 0.0000),
    vec4(-0.6041, -0.4376, 0.6660, 0.0000),
    vec4(0.0009, -0.6983, -0.7158, 0.0000),
    vec4(-0.2054, 0.1049, 0.9730, 0.0000),
    vec4(0.6791, -0.1330, 0.7219, 0.0000),
    vec4(0.5587, -0.7958, 0.2338, 0.0000),
    vec4(-0.8293, 0.4653, -0.3094, 0.0000),
    vec4(-0.1252, 0.7826, -0.6098, 0.0000),
    vec4(-0.5253, 0.8178, -0.2348, 0.0000),
    vec4(-0.4495, -0.0901, -0.8887, 0.0000),
    vec4(-0.8948, -0.3001, 0.3305, 0.0000),
    vec4(-0.6230, -0.5319, -0.5735, 0.0000),
    vec4(-0.3486, -0.0990, 0.9320, 0.0000),
    vec4(0.1814, 0.5446, -0.8189, 0.0000),
    vec4(-0.5766, 0.4495, -0.6823, 0.0000),
    vec4(-0.6117, -0.3574, -0.7058, 0.0000),
    vec4(-0.6711, -0.7362, -0.0873, 0.0000),
    vec4(0.4014, -0.6195, -0.6746, 0.0000),
    vec4(0.3167, -0.8831, 0.3461, 0.0000),
    vec4(0.5305, -0.0033, -0.8477, 0.0000),
    vec4(0.0713, 0.9205, 0.3842, 0.0000),
    vec4(0.4199, -0.6517, 0.6316, 0.0000),
    vec4(0.0044, 0.6338, 0.7735, 0.0000),
    vec4(-0.0996, 0.5615, -0.8214, 0.0000),
    vec4(0.7404, 0.5374, -0.4036, 0.0000),
    vec4(0.7109, 0.7000, 0.0675, 0.0000),
    vec4(0.5963, 0.0959, -0.7970, 0.0000),
    vec4(0.0171, 0.8357, 0.5489, 0.0000),
    vec4(0.5390, 0.7638, 0.3550, 0.0000),
    vec4(-0.9857, -0.1083, 0.1294, 0.0000),
    vec4(0.0474, -0.9941, 0.0973, 0.0000),
    vec4(-0.6146, 0.6077, 0.5030, 0.0000),
    vec4(0.1841, -0.9342, 0.3057, 0.0000),
    vec4(-0.9721, -0.2345, -0.0046, 0.0000),
    vec4(0.3296, 0.8431, -0.4249, 0.0000),
    vec4(-0.6883, -0.3305, 0.6457, 0.0000),
    vec4(-0.2294, -0.4088, 0.8833, 0.0000),
    vec4(-0.7671, 0.1249, -0.6293, 0.0000),
    vec4(-0.6832, -0.6061, 0.4073, 0.0000),
    vec4(0.5189, 0.6415, -0.5650, 0.0000),
    vec4(0.6458, -0.4373, 0.6259, 0.0000),
    vec4(0.7107, -0.1808, -0.6799, 0.0000),
    vec4(-0.3510, -0.6983, 0.6238, 0.0000),
    vec4(-0.7280, 0.1924, 0.6580, 0.0000),
    vec4(-0.6460, 0.7058, 0.2908, 0.0000),
    vec4(-0.8157, -0.4794, -0.3237, 0.0000),
    vec4(-0.6803, -0.0757, 0.7290, 0.0000),
    vec4(-0.1603, -0.7526, -0.6387, 0.0000),
    vec4(-0.6408, 0.1400, 0.7549, 0.0000),
    vec4(-0.4353, -0.6259, -0.6472, 0.0000),
    vec4(0.5218, 0.8466, -0.1044, 0.0000),
    vec4(0.5672, -0.5317, 0.6289, 0.0000),
    vec4(0.6977, 0.6814, -0.2212, 0.0000),
    vec4(0.5921, 0.0553, -0.8040, 0.0000),
    vec4(-0.5309, 0.8156, 0.2302, 0.0000),
    vec4(-0.0842, 0.7846, -0.6143, 0.0000),
    vec4(-0.7280, -0.2952, -0.6188, 0.0000),
    vec4(-0.2901, -0.6370, -0.7142, 0.0000),
    vec4(0.9439, -0.1084, -0.3121, 0.0000),
    vec4(0.2151, 0.1761, 0.9606, 0.0000),
    vec4(-0.9804, -0.0671, -0.1851, 0.0000),
    vec4(0.0466, -0.1335, -0.9899, 0.0000),
    vec4(-0.3668, -0.0189, -0.9301, 0.0000),
    vec4(-0.4480, 0.8216, -0.3525, 0.0000),
    vec4(0.2536, -0.5505, 0.7954, 0.0000),
    vec4(-0.6280, 0.6439, -0.4371, 0.0000),
    vec4(0.6143, 0.6177, 0.4909, 0.0000),
    vec4(0.8212, 0.4258, -0.3800, 0.0000),
    vec4(-0.2537, -0.3749, 0.8917, 0.0000),
    vec4(-0.0466, 0.6658, -0.7447, 0.0000),
    vec4(-0.8876, -0.0072, -0.4606, 0.0000),
    vec4(-0.2564, -0.7359, -0.6266, 0.0000),
    vec4(0.6018, 0.4833, -0.6358, 0.0000),

];

const _PERMS: [usize; 8] = [
    1, 4, 5, 0, 7, 3, 6, 2,
];

fn grad1(a: f32) -> f32 {
    grad2(a, 0.0).x
}

fn grad2(x: f32, y: f32) -> Vec4 {
    GRADIENTS[(83.0 * x + 29.0 * y) as usize & 0xFF]
}

pub fn perlin_noise1(x: f32) -> f32 {
    let (x0, x) = (x.floor(), x.fract());

    let g0 = grad1(x0);
    let g1 = grad1(x0 + 1.0);


    lerp(smoothstep(x), g0, g1)
}

fn dot(x0: f32, y0: f32, x: f32, y: f32) -> f32 {
    let g = grad2(x0, y0);
    let (dx, dy) = (x - x0, y - y0);
    g.x * dx + g.y * dy
}

pub fn perlin_noise2(x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (x1, y1) = (x0 + 1.0, y0 + 1.0);

    let g00 = dot(x0, y0, x, y);
    let g01 = dot(x0, y1, x, y);
    let g10 = dot(x1, y0, x, y);
    let g11 = dot(x1, y1, x, y);

    let x = smoothstep(x.fract());
    let y = smoothstep(y.fract());

    lerp(y,
        lerp(x, g00, g10),
        lerp(x, g01, g11)
    )
}

pub fn perlin_noise(pt: Vec4) -> f32 {
    perlin_noise2(pt.x, pt.y)
}

pub fn fractal_noise<S>(o: u32, f: f32, a: f32, source: S)
    -> impl Signal<Vec4, R=f32>
where
    S: Signal<Vec4, R=f32> + Copy,
{
    move |v| {
        let mut freq = 1.0;
        let mut amp = 1.0;
        let mut res = 0.0;
        for _ in 0..o {
            res += amp * source.sample(freq * v);
            freq *= f;
            amp *= a;
        }
        res
    }
}

pub fn vector_noise2(x: f32, y: f32) -> Vec4 {
    let (x0, y0) = (x.floor(), y.floor());
    let (x1, y1) = (x0 + 1.0, y0 + 1.0);

    let g00 = grad2(x0, y0);
    let g01 = grad2(x0, y1);
    let g10 = grad2(x1, y0);
    let g11 = grad2(x1, y1);

    let x = smoothstep(x.fract());
    let y = smoothstep(y.fract());

    lerp(y,
         lerp(x, g00, g10),
         lerp(x, g01, g11)
    )
}

pub fn vector_noise(v: Vec4) -> Vec4 {
    vector_noise2(v.x, v.y)
}

#[cfg(test)]
mod tests {
    use math::rand::Random;
    use math::vec::UnitDir;

    use super::*;

    #[test]
    fn sadfg() {
        let mut r = Random::new();

        for v in r.iter(UnitDir).take(256) {
            eprintln!("vec4{:.4},", v);
        }
    }


    #[test]
    fn perlin1() {

        for i in 0..=40 {
            let x = i as f32 / 4.0;
            eprintln!("{:2.1} {:2.2}", x, perlin_noise1(x));
        }
    }

    #[test]
    fn perlin2() {
        for i in 0..=20 {
            for j in 0..=20 {
                let x = i as f32 / 4.0;
                let y = j as f32 / 4.0;

                let n = perlin_noise2(x, y);
                let i = (n * 96.0) as i8;
                eprint!("{:4}", i);
            }
            eprintln!();
        }
    }
}

/*

f = ax^3 + bx^2 + cx + d
f' = 3ax^2 + 2bx + c

f(0) = 0
f(1) = 0
f'(0) = s
f'(1) = t

c = s
3a + 2b + c = t

d = 0
a + b + c + d = 0
a + b + c = 0

3a + 2b + s = t
a + b + s = 0

3a + 2b = t-s
a + b = -s

b = -s - a

3a + 2(-s-a) = t-s

3a - 2s - 2a = t-s

a = t+s
b = -s-t-s = -2s-t
c = s
d = 0

f = (t+s)x^3 - (2s+t)x^2 + sx

 */
