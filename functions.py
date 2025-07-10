"""
[(4087, 296, 87, 24),
 (4231, 294, 181, 24),
 (5265, 292, 65, 11),
 (4175, 292, 62, 28),
 (4420, 291, 52, 27),
 (3797, 290, 68, 13),
 (6581, 285, 524, 20),
 (6417, 284, 103, 21),
 (7196, 282, 1223, 38),
 (1991, 279, 44, 12),
 (4624, 277, 104, 19),
 (4497, 276, 63, 41),
 (4025, 276, 57, 27),
 (3533, 275, 259, 41),
 (4766, 274, 140, 21),
 (4371, 273, 85, 9),
 (2422, 271, 152, 18),
 (2476, 270, 1032, 43),
 (3858, 269, 46, 15),
 (5444, 268, 38, 21),
 (3768, 268, 81, 18),
 (2780, 268, 47, 14),
 (5753, 265, 98, 17),
 (4953, 265, 338, 33),
 (5523, 264, 111, 20),
 (5307, 260, 109, 19),
 (5947, 254, 102, 31),
 (0, 250, 25, 78),
 (8427, 244, 157, 135)]
"""

def group_nearby_boxes(boxes, x_distance_threshold=50, y_distance_threshold=50, overlap_threshold=0.3):
    """
    Group bounding boxes that are close to each other.
    
    Args:
        boxes: List of tuples (x, y, w, h) representing bounding boxes
        distance_threshold: Maximum distance between box centers to consider them nearby (default: 50)
        overlap_threshold: Minimum overlap ratio to consider boxes as overlapping (default: 0.3)
    
    Returns:
        List of grouped boxes, where each group is a list of box indices
    """
    if not boxes:
        return []
    
    def calculate_box_center(box):
        """Calculate center point of a box (x, y, w, h)"""
        x, y, w, h = box
        return (x + w//2, y + h//2)
    
    def calculate_box_area(box):
        """Calculate area of a box"""
        x, y, w, h = box
        return w * h
    
    def boxes_overlap(box1, box2):
        """Check if two boxes overlap significantly"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        smaller_area = min(area1, area2)
        
        return intersection_area / smaller_area > overlap_threshold
    
    def boxes_close(box1, box2):
        """Check if two boxes are close to each other based on boundary distances"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate boundary distances
        # X-axis distance (horizontal)
        if x1 + w1 < x2:  # box1 is to the left of box2
            x_distance = x2 - (x1 + w1)
        elif x2 + w2 < x1:  # box2 is to the left of box1
            x_distance = x1 - (x2 + w2)
        else:  # boxes overlap in x-axis
            x_distance = 0
        
        # Y-axis distance (vertical)
        if y1 + h1 < y2:  # box1 is above box2
            y_distance = y2 - (y1 + h1)
        elif y2 + h2 < y1:  # box2 is above box1
            y_distance = y1 - (y2 + h2)
        else:  # boxes overlap in y-axis
            y_distance = 0
        
        # Check if boxes are close in BOTH axes (AND condition)
        # Only group if boxes are close in both X and Y directions
        return x_distance <= x_distance_threshold and y_distance <= y_distance_threshold
    
    def should_group_boxes(box1, box2):
        """Determine if two boxes should be grouped together"""
        return boxes_overlap(box1, box2) or boxes_close(box1, box2)
    
    # Initialize groups
    groups = []
    used_indices = set()
    
    for i, box in enumerate(boxes):
        if i in used_indices:
            continue
        
        # Start a new group
        current_group = [i]
        used_indices.add(i)
        
        # Find all boxes that should be grouped with this one
        changed = True
        while changed:
            changed = False
            for j, other_box in enumerate(boxes):
                if j in used_indices:
                    continue
                
                # Check if any box in current group should be grouped with this box
                should_add = False
                for group_idx in current_group:
                    if should_group_boxes(boxes[group_idx], other_box):
                        should_add = True
                        break
                
                if should_add:
                    current_group.append(j)
                    used_indices.add(j)
                    changed = True
        
        groups.append(current_group)
    print(groups)
    return groups

def merge_grouped_boxes(boxes, groups):
    """
    Merge grouped boxes into single bounding boxes.
    
    Args:
        boxes: List of tuples (x, y, w, h) representing bounding boxes
        groups: List of groups, where each group is a list of box indices
    
    Returns:
        List of merged boxes as tuples (x, y, w, h)
    """
    merged_boxes = []
    
    for group in groups:
        if not group:
            continue
        
        # Get all boxes in this group
        group_boxes = [boxes[i] for i in group]
        
        # Calculate the bounding box that encompasses all boxes in the group
        min_x = min(box[0] for box in group_boxes)
        min_y = min(box[1] for box in group_boxes)
        max_x = max(box[0] + box[2] for box in group_boxes)
        max_y = max(box[1] + box[3] for box in group_boxes)
        
        # Create merged box
        merged_box = (min_x, min_y, max_x - min_x, max_y - min_y)
        merged_boxes.append(merged_box)
    
    return merged_boxes

def group_and_merge_boxes(boxes, x_distance_threshold=50, y_distance_threshold=50, overlap_threshold=0.3):
    """
    Convenience function that groups nearby boxes and merges them.
    
    Args:
        boxes: List of tuples (x, y, w, h) representing bounding boxes
        distance_threshold: Maximum distance between box centers to consider them nearby
        overlap_threshold: Minimum overlap ratio to consider boxes as overlapping
    
    Returns:
        List of merged boxes as tuples (x, y, w, h)
    """
    groups = group_nearby_boxes(boxes, x_distance_threshold, y_distance_threshold, overlap_threshold)
    return merge_grouped_boxes(boxes, groups)

# Example usage:
if __name__ == "__main__":
    # Example boxes from your data
    example_boxes = [
        (4087, 296, 87, 24),
        (4231, 294, 181, 24),
        (5265, 292, 65, 11),
        (4175, 292, 62, 28),
        (4420, 291, 52, 27),
        (3797, 290, 68, 13),
        (6581, 285, 524, 20),
        (6417, 284, 103, 21),
        (7196, 282, 1223, 38),
        (1991, 279, 44, 12),
        (4624, 277, 104, 19),
        (4497, 276, 63, 41),
        (4025, 276, 57, 27),
        (3533, 275, 259, 41),
        (4766, 274, 140, 21),
        (4371, 273, 85, 9),
        (2422, 271, 152, 18),
        (2476, 270, 1032, 43),
        (3858, 269, 46, 15),
        (5444, 268, 38, 21),
        (3768, 268, 81, 18),
        (2780, 268, 47, 14),
        (5753, 265, 98, 17),
        (4953, 265, 338, 33),
        (5523, 264, 111, 20),
        (5307, 260, 109, 19),
        (5947, 254, 102, 31),
        (0, 250, 25, 78),
        (8427, 244, 157, 135)
    ]
    
    print("Original boxes:", len(example_boxes))
    
    # Group nearby boxes
    groups = group_nearby_boxes(example_boxes, distance_threshold=100, overlap_threshold=0.1)
    print(f"Found {len(groups)} groups:")
    for i, group in enumerate(groups):
        print(f"  Group {i}: {len(group)} boxes - indices: {group}")
    
    # Merge grouped boxes
    merged_boxes = merge_grouped_boxes(example_boxes, groups)
    print(f"\nMerged into {len(merged_boxes)} boxes:")
    for i, box in enumerate(merged_boxes):
        print(f"  Box {i}: {box}")
    
    # Or use the convenience function
    final_boxes = group_and_merge_boxes(example_boxes, distance_threshold=100, overlap_threshold=0.1)
    print(f"\nFinal result: {len(final_boxes)} merged boxes")
