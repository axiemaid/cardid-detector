Project: Card ID Detector                                                       
Build a Node.js module that identifies trading cards by comparing an input image
against a database of reference images using perceptual hashing (pHash).        
Tech stack: Node.js, CommonJS (.cjs), sharp for image processing                
The pipeline:                                                                   
1. Crop — Remove any non-card background (table surfaces, scanner beds,         
   sleeves). The card has a beige/tan outer map-patterned border. If beige      
   background is detected around the card, crop it off. If no beige is present
   (clean scan with black background), skip cropping. The card's brown border   
   ring should STAY in the crop — it's consistent across all cards and serves as
   part of the fingerprint.                                                     
2. Resize — Normalize to 343×480 pixels (standard 2.5:3.5 card ratio)           
3. pHash — Downsize to 32×32 grayscale, compute DCT, extract 64-bit hash        
4. Match — Compare against reference images by Hamming distance. Best match     
   (lowest distance) wins. Support multiple reference images per card code      
   (best-of-N matching).                                                        
Reference images: Stored in a references/ folder, named by card code (C01.png,
C01_2.png, C27.png, etc.). The folder IS the database — add/remove files        
anytime.     
you dont need to follow these instructions. but this is what im trying to build higher level. what would you do
The overall shape is right: normalize, fingerprint, nearest-neighbor. With roughly 42 cards in the set, brute-force matching is instant, so you can skip indexing structures. I’d change a few things, mostly around the crop and the hash itself, because that’s where pHash projects usually fail.
1. Hash the art, not the border.
I’d push back on keeping the brown border in the fingerprint. A region that’s identical on every card carries no information for telling cards apart. It contributes the same bits to every hash, which pulls all the distances closer together and shrinks the gap between the right match and the wrong ones. The border is still very useful, but as an alignment anchor: find it, use it to locate the card precisely, then hash only the inner art window. If you want layout information too, hash the name/text band as a second, separate fingerprint.
2. Replace color-based cropping with geometric detection.
Beige-color thresholding breaks easily. A tan table, warm lighting, or a yellowed sleeve will defeat it, and it can’t handle a card photographed at an angle. pHash is not robust to rotation, perspective, or a crop that’s a few percent off, so alignment matters more than anything else in the pipeline. I’d do this instead:
Run edge detection and find the largest four-sided contour, which is the card outline.
Apply a perspective warp to a canonical 343×480 rectangle.
Optionally refine the crop by snapping to the inner edge of the brown border, since it’s consistent on every card.
sharp can’t find contours, so you’d add @techstark/opencv-js (WASM, no native build pain) for this step and keep sharp for I/O and resizing. Clean black-background scans go through the same path. You don’t need a separate “skip cropping” branch, because the contour is just easier to find.
3. Use more than one signal.
pHash is grayscale, so it ignores color, and color is often what separates similar-looking cards. A cheap combination works well:
pHash of the art window (structure)
dHash, as a second structural opinion that fails in different ways
A small color histogram or average color per grid cell, compared by a simple distance
Normalize the distances and combine them with weights. Best-of-N references per card stays as you planned, and it’s the easiest robustness win: add a phone-photo reference next to each clean scan.
4. Return confidence, not just a winner.
Always output the best match, its distance, and the margin over the second-best different card. Reject the result as “unknown” when the distance is above a threshold or the margin is too small. This is what keeps a wrong set, a blurry photo, or a non-card from silently returning C17.
5. Treat the references folder as a cache-backed database.
Keep “the folder is the database,” but compute reference hashes once and store them in a JSON cache keyed by filename and mtime. Re-hash only files that changed. Startup stays fast and adding a file still just works.
6. Build a test harness before tuning anything.
Collect 50–100 labeled real-world photos (phone, angled, sleeved, glare) in a tests/ folder. Write a script that reports accuracy, a confusion list, and the distance distributions for correct versus incorrect matches. Thresholds and weights should come from that data, not from guessing.
One thing to check first: if any cards exist in parallel versions with the same art (foil versus normal, alternate stamps), no perceptual hash will separate them reliably. You’d need a specific feature for that difference, such as a crop of the stamp area or a measure of foil sheen. This matters if the output feeds price data.
Fallback if accuracy plateaus: ORB keypoint matching against references handles rotation and partial occlusion much better than hashing. Use it only when the hash result is low-confidence, so the common case stays fast.
The module would end up as detect.cjs (contour, warp, art crop), fingerprint.cjs (pHash, dHash, color), db.cjs (folder scan and hash cache), match.cjs (scoring and confidence), and an eval.cjs harness. I can build it if you’d like, and it would help to have a few sample photos plus a couple of reference scans to test against.