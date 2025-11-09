import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
import os 

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

#Spongebob = input_image = load_image("https://static.wikia.nocookie.net/spongebob/images/b/be/The_Sponge_Who_Could_Fly_094.png/revision/latest/scale-to-width-down/1422?cb=20191116013524")
#input_image = load_image('https://sacompassion.net/wp-content/uploads/2016/08/black-dot.jpg')
#input_image = load_image('https://muralsyourway.vtexassets.com/arquivos/ids/274926-825-auto?width=825&height=auto&aspect=true')
#input_image = load_image('https://www.regencychess.co.uk/images/how-to-set-up-a-chessboard/how-to-set-up-a-chessboard-7.jpg')
#input_image = load_image('https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEhk3k3ViUY16YjFLwGrbJZMyae-7UnvdJdIEyjtks4bUwYDhuTiRjV2sYfurYpufelj1xxcpNAwcz3VQeHYmw_nMjBgRNNsR1s4prLWmCMK3ElvArXzJ34oRQfFQe7VsIKsyu7EFszsp2Q/s1600/Transparent-Lodi.jpg')
#input_image = load_image('https://helloartsy.com/wp-content/uploads/kids/places/how-to-draw-a-windmill/how-to-draw-a-windmill-step-9.jpg')
#input_image = load_image('https://cdn.mos.cms.futurecdn.net/qRPNjNcZYZHMh9cMLQGSTY.jpg')
#input_image = load_image('https://media.istockphoto.com/id/1329335620/photo/aerial-view-of-sailing-luxury-yacht-at-opened-sea-at-sunny-day-in-croatia.jpg?s=612x612&w=0&k=20&c=d6DNf5lOEJFegq2YbTJEhA8n5Qj2aMD9u_CBAH3pHtc=')
input_image = load_image('https://www.hungryones.com/wp-content/uploads/2016/09/statue.liberty.1527.jpg')


liberty_prompt = [
    # Frame 2
    "Continuation of the same scene. The camera moves slightly closer to the distant monument across the calm sea, as if a drone is gliding forward. "
    "The monument appears marginally larger but remains centered on the horizon. The light reflections on the water stay identical, and the overall lighting and sky remain unchanged.",

    # Frame 3
    "Next frame continuing seamlessly. The camera moves closer again toward the monument, which appears slightly more detailed, while the horizon, sunlight reflections, and soft haze remain perfectly consistent. "
    "Keep the same composition, tone, and atmosphere as before.",

    # Frame 4
    "Next frame continuing the same smooth motion. The drone approaches halfway toward the monument’s original distance, showing slightly more detail near its base while the background and water texture remain stable. "
    "Lighting direction and sky gradient are unchanged.",

    # Frame 5
    "Next frame continuing the forward motion. The monument now occupies a larger portion of the frame, while the ocean’s ripples and reflections remain coherent. "
    "Maintain identical color tones, sunlight direction, and horizon alignment.",

    # Frame 6 (final)
    "Final frame completing the cinematic camera approach. The drone reaches a closer, more detailed view of the monument standing at the horizon, surrounded by the same calm water and golden sunlight. "
    "The scene remains consistent in lighting, composition, and atmosphere."
]




output_dir = "/restricted/projectnb/cs599dg/onur/experiment1_2/generated_frames/liberty"
os.makedirs(output_dir, exist_ok=True)
# === Generation loop ===
for i, prompt in enumerate(liberty_prompt, start=2):
    out_path = os.path.join(output_dir, f"{i:04d}.png")
    print(prompt)
    result = pipe(image=input_image, prompt=prompt)
    next_frame = result.images[0]
    next_frame.save(out_path)
    print(f"Saved: {out_path}")

    input_image = next_frame




    '''
    prompts = [
    "Generate the next frame continuing from the given image. SpongeBob lifts his left foot off the ground, knee slightly forward; right foot stays planted. Keep the same underwater blue background with flower shapes, same road, lighting, and camera framing; preserve art style and character identity.",

    "Generate the next frame: left knee advances forward; right arm swings forward, left arm swings back, small upward body bob. Background, road, palette, and perspective unchanged.",

    "Generate the next frame: left foot swings to mid-stride ahead of the body, not yet touching the ground; torso leans slightly forward; right heel still down. Keep identical background and framing; no new objects.",

    "Generate the next frame: left heel touches down ahead of the right foot, toes up; right foot begins to lift at the heel. Maintain exact background layout and camera.",

    "Generate the next frame: weight transfers onto the left foot; right heel rises more; left knee straightens slightly; arms continue natural swing. Background colors and shapes remain fixed.",

    "Generate the next frame: right foot lifts off the ground; left foot is flat and stable bearing weight; small downward bob of the body. Preserve the same road and flower positions.",

    "Generate the next frame: right foot passes the left at mid-air, knee forward; left arm swings forward, right arm back. Keep the same composition and lighting.",

    "Generate the next frame: right heel strikes the ground ahead, completing the first step; left foot starts to lift at the heel. Background and camera unchanged; maintain cel-shaded cartoon look.",

    "Generate the next frame: weight shifts onto the right foot; left heel rises more; mild torso rise. Keep identical underwater sky, flower shapes, and road position.",

    "Generate the next frame: left foot clears the ground; right foot is flat and stable; arms continue opposite swing. Match framing and scene exactly.",

    "Generate the next frame: left foot passes the right in mid-air, crossover position; small forward lean; right arm forward, left arm back. Background and static elements remain locked.",

    "Generate the next frame: left heel touches down ahead, starting the second step; right foot begins to lift at heel. Keep the same perspective and palette; no new details.",

    "Generate the next frame: weight transfers onto the left; body dips slightly from impact; right toes still touching. Maintain fixed background and camera.",

    "Generate the next frame: right foot lifts fully, preparing swing; arms approach neutral mid-swing; torso returns upright. Background, road, lighting identical.",

    "Generate the next frame: settle into a stable stance after two steps — left foot planted ahead, right foot mid-air beginning the next cycle; minimal motion to make a clean loop point. Keep exact background alignment and framing."
]

apple_prompt = [
    # Frame 2
    "Generate the next frame continuing seamlessly from the given image. "
    "The same apple remains in the exact position and lighting, but it begins to ripen slightly — the red tones deepen a little, and the surface becomes glossier. "
    "Keep the white background, the same camera angle, shadows, and reflections identical to the reference image. "
    "No new elements appear; the apple’s shape and stem remain exactly the same.",

    # Frame 3
    "Generate the next frame following directly from the previous one. "
    "The same apple continues to ripen; its color shifts slightly toward a deeper crimson red with warmer highlights. "
    "The texture remains smooth and natural, with subtle reflections in the same lighting direction. "
    "Preserve identical framing, white background, and camera perspective.",

    # Frame 4
    "Generate the next frame continuing the ripening process. "
    "The apple now shows a more vivid, saturated red hue, with faint golden undertones near the base. "
    "The light reflections are stronger but still from the same direction, and the shape of the apple and stem are unchanged. "
    "Maintain a consistent white background and identical camera setup.",

    # Frame 5
    "Generate the next frame continuing from the previous one. "
    "The apple’s red color deepens further, now appearing at full ripeness — rich crimson with warm yellow reflections. "
    "Slightly enhance the surface shine while keeping all geometry, camera, and lighting identical. "
    "No changes to the background or composition.",

    # Frame 6 (final)
    "Generate the final frame continuing seamlessly. "
    "The same apple remains in the same place, now appearing freshly polished and perfectly ripe — uniform deep red, smooth glossy skin. "
    "The background remains pure white with no color cast or shadows altered. "
    "Do not introduce any new objects or variations; the scene stays identical except for the apple’s final ripened appearance."
]

windmill_prompt = [
    # Frame 2
    "Generate the next frame continuing seamlessly from the given image. "
    "The same windmill remains in the exact same position, lighting, and framing, but its four blades have rotated slightly clockwise — about 15 degrees. "
    "The building structure, roof color, and texture remain identical. "
    "Preserve the same background and overall composition without introducing any new elements or artifacts.",

    # Frame 3
    "Generate the next frame following directly from the previous one. "
    "The windmill blades continue rotating clockwise by another 15 degrees. "
    "The wooden frames of the blades retain their geometry and thickness, and the hub stays perfectly centered. "
    "Keep the windmill tower, colors, and background completely unchanged.",

    # Frame 4
    "Generate the next frame continuing the clockwise rotation. "
    "The windmill blades now appear rotated around 45 degrees from the original starting position. "
    "All lighting, shadows, outlines, and building proportions remain exactly the same. "
    "Ensure there are no distortions or perspective changes in the blades or structure.",

    # Frame 5
    "Generate the next frame continuing seamlessly. "
    "The windmill blades rotate another 15 degrees clockwise, maintaining perfect alignment and symmetry around the hub. "
    "The tower, windows, and background stay identical to the previous frames, with no color or shape drift.",

    # Frame 6 (final)
    "Generate the next frame completing a partial rotation. "
    "The windmill blades have rotated approximately 90 degrees clockwise compared to the initial frame. "
    "The tower, roof, windows, and lighting are unchanged, and the background remains a uniform white texture. "
    "Do not alter the composition, perspective, or add any new features — only the blades’ rotation continues."
]

prompts = [
    # Frame 2
    "Generate the next frame continuing seamlessly from the given image. "
    "The leftmost steel ball swings inward along its string, halfway toward the center. "
    "The four right balls remain motionless, perfectly aligned and touching. "
    "Keep the same chrome reflections, white background, and stand geometry identical.",

    # Frame 3
    "Generate the next frame following directly from the previous one. "
    "The leftmost ball now contacts the second ball at the center line. "
    "The three middle balls stay almost still; the rightmost ball begins to lift slightly outward. "
    "Maintain identical camera angle, lighting, and reflections.",

    # Frame 4
    "Generate the next frame continuing the motion. "
    "The leftmost ball stops at the center while the rightmost ball swings outward on its string to the opposite side. "
    "The other four balls remain nearly vertical. "
    "Keep all materials, stand, and background unchanged.",

    # Frame 5
    "Generate the next frame. "
    "The rightmost ball reaches its highest outward point, string taut, while the other four stay aligned. "
    "Reflections and lighting are exactly the same; no background or stand movement.",

    # Frame 6 (final)
    "Generate the next frame completing the energy transfer. "
    "The rightmost ball swings back toward the row and strikes; the leftmost ball begins lifting outward slightly. "
    "All geometry, reflections, and background remain perfectly constant."
]


zoom_prompt =  [
    # Frame 2
    "Continuation of the same scene. The camera moves slightly closer to the sailboat across the calm sea, as if a drone is flying forward. "
    "The boat appears marginally larger but stays centered in frame, the ocean surface and ripples remain identical. "
    "Lighting and sky color are unchanged; maintain the same horizon and reflections.",

    # Frame 3
    "Next frame continuing seamlessly. The drone camera has moved closer to the sailboat again — the vessel appears noticeably larger and clearer, while the sea and sky remain perfectly consistent. "
    "The water texture, horizon, and light direction stay the same. No new elements or waves appear.",

    # Frame 4
    "Next frame. The camera continues its steady approach, now halfway to the boat’s initial distance. "
    "The boat occupies more of the frame, showing slightly more detail on the deck, while maintaining identical lighting, sea texture, and horizon position.",

    # Frame 5
    "Next frame continuing the forward motion. The sailboat is now much closer, filling a larger portion of the frame, while the surrounding ocean and lighting remain perfectly stable. "
    "Reflections and wake are consistent with previous frames; background horizon remains fixed.",

    # Frame 6 (final)
    "Final frame completing the smooth camera approach. The sailboat is now close-up and detailed, centered on the same horizon line with identical ocean lighting. "
    "The sea and sky gradients are unchanged, and no new objects or distortions appear. "
    "Maintain cinematic continuity and perfect color consistency."
]
'''