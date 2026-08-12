/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::collections::{HashMap, HashSet};

use app_units::Au;
use euclid::Rect;
use paint_api::largest_contentful_paint_candidate::{LCPCandidate, LCPCandidateID};
use servo_geometry::FastLayoutTransform;
use servo_url::ServoUrl;
use style::dom::OpaqueNode;
use webrender_api::units::{LayoutRect, LayoutSize};

use crate::fragment_tree::Tag;
use crate::query::transform_f32_rectangle;

/// An image LCP candidate collected during paint traversal.
/// Corresponds to a `pending image record` in the spec.
struct ImageRecord {
    /// The image element this record belongs to.
    tag: Tag,
    /// The image rect (adjusted for object-fit/object-position).
    bounds: LayoutRect,
    /// The element's content box.
    clip_rect: LayoutRect,
    /// Cumulative transform to root space, computed at collection time.
    transform: FastLayoutTransform,
    /// The image URL. `None` for background images.
    url: Option<ServoUrl>,
    /// Intrinsic width, used for upscaling normalization.
    natural_width: Option<Au>,
    /// Intrinsic height, used for upscaling normalization.
    natural_height: Option<Au>,
}

/// A text LCP candidate accumulated during paint traversal.
/// Corresponds to a `text element` in the spec's `paintedTextNodes`.
struct TextRecord {
    /// The containing element this text belongs to.
    tag: Tag,
    /// All text fragment rects (world space). Unioned at compute time,
    /// matching the spec's "union of the border boxes of all Text nodes".
    rects: Vec<LayoutRect>,
}

pub(crate) struct PaintTimingHandler {
    /// The rect of viewport.
    viewport_rect: LayoutRect,
    /// The document's largest contentful paint size
    lcp_size: f32,
    /// Counter for generating unique LCP candidate UUIDs.
    lcp_next_uuid: u64,
    /// The LCP candidate, it may be a image or text.
    lcp_candidate: Option<LCPCandidate>,
    /// The DOM node for the current LCP candidate. Only used in ReflowResult
    current_lcp_node: Option<OpaqueNode>,
    /// Flag to indicate if there is an update to LCP candidate.
    /// This is used to avoid sending duplicate LCP candidates to `Paint`.
    lcp_candidate_updated: bool,
    /// Nodes whose image LCP candidates have already been reported.
    /// Corresponds to `paintedImages` in the spec.
    reported_image_nodes: HashSet<OpaqueNode>,
    /// Nodes whose text LCP candidates have already been reported.
    /// Corresponds to `paintedTextNodes` in the spec.
    reported_text_nodes: HashSet<OpaqueNode>,
    /// Image records collected during the current paint traversal.
    painted_images: Vec<ImageRecord>,
    /// Text records accumulated during the current paint traversal,
    /// keyed by containing element node.
    painted_text_nodes: HashMap<OpaqueNode, TextRecord>,
}

impl PaintTimingHandler {
    pub(crate) fn new(viewport_size: LayoutSize) -> Self {
        Self {
            lcp_size: 0.0,
            lcp_next_uuid: 0,
            current_lcp_node: None,
            lcp_candidate: None,
            lcp_candidate_updated: false,
            viewport_rect: LayoutRect::from_size(viewport_size),
            reported_image_nodes: HashSet::new(),
            reported_text_nodes: HashSet::new(),
            painted_images: Vec::new(),
            painted_text_nodes: HashMap::new(),
        }
    }

    // Returns true if has non-zero width and height values.
    pub(crate) fn check_bounding_rect(&self, bounds: LayoutRect, clip_rect: LayoutRect) -> bool {
        let clipped_rect = bounds
            .intersection(&clip_rect)
            .unwrap_or(LayoutRect::zero())
            .to_rect();

        let bounding_rect = clipped_rect
            .intersection(&self.viewport_rect.to_rect().cast_unit())
            .unwrap_or(Rect::zero());

        !bounding_rect.is_empty()
    }

    /// <https://www.w3.org/TR/largest-contentful-paint/#sec-effective-visual-size>
    fn effective_visual_size(
        &self,
        bounds: LayoutRect,
        clip_rect: LayoutRect,
        intersection_rect: LayoutRect,
        transform: FastLayoutTransform,
        natural_width: Option<Au>,
        natural_height: Option<Au>,
    ) -> Option<f32> {
        // Step 1. Let width be intersectionRect's width, rounded up to the
        // nearest integer.
        // Step 2. Let height be intersectionRect's height, rounded up to the
        // nearest integer.
        // Step 3. Let size be width * height.
        let size = intersection_rect.area();

        // Step 4. Let root be document's browsing context's top-level browsing
        // context's active document.
        // Note: This is not needed as we already have the viewport rect.

        // Step 5. Let rootWidth be root's visual viewport's width,
        // excluding any scrollbars.
        // Step 6. Let rootHeight be root's visual viewport's height excluding
        // any scrollbars.
        // Step 7. If size is equal to rootWidth times rootHeight, return null.
        if size >= self.viewport_rect.area() {
            return None;
        }

        // Step 8: If imageRequest is not null, run the following steps to
        // adjust for image position and upscaling:
        // Note: This is handled by check for [showing_broken_image_icon] earlier

        // TODO Step 8.1: If imageRequest's response's content length in bytes
        // is less than size * 0.004, then return null. (Not Implemented)

        // Step 8.2: Let concreteDimensions be imageRequest's concrete object
        // size within element.
        // Step 8.3: Let visibleDimensions be concreteDimensions, adjusted for
        // positioning by object-position or background-position and element's
        // content box.
        // Note: bounds are already adjusted for positioning and content box
        let visible_dimensions = bounds
            .intersection(&clip_rect)
            .unwrap_or(LayoutRect::zero());

        // Step 8.4: Let clientContentRect be the smallest DOMRectReadOnly
        // containing visibleDimensions with element's transforms applied.
        let client_content_rect =
            transform_f32_rectangle(visible_dimensions.to_rect(), transform).unwrap_or_default();

        // Step 8.5: Let intersectingClientContentRect be the intersection of
        // clientContentRect with intersectionRect.
        let intersecting_client_content_rect = client_content_rect
            .intersection(&intersection_rect.to_rect())
            .unwrap_or(Rect::zero());

        // Step 8.6: Set width to intersectingClientContentRect's width,
        // rounded up to the nearest integer.
        // Step 8.7: Set height to intersectingClientContentRect's height,
        // rounded up to the nearest integer.
        // Step 8.8: Set size to width * height.
        let mut size = intersecting_client_content_rect.area();

        // Step 8.9: Let naturalArea be imageRequest's natural width * imageRequest's natural height.
        if let (Some(natural_width), Some(natural_height)) = (natural_width, natural_height) {
            let natural_area = natural_width.to_f32_px() * natural_height.to_f32_px();

            // Step 8.10: If naturalArea is 0, then return null.
            if natural_area == 0.0 {
                return None;
            }
            // Step 8.11: Let boundingClientArea be clientContentRect's width *
            // clientContentRect's height.
            let bounding_client_area = client_content_rect.width() * client_content_rect.height();

            // Step 8.12: Let scaleFactor be boundingClientArea / naturalArea.
            let scale_factor = bounding_client_area / natural_area;

            // Step 8.13: If scaleFactor is greater than 1, then divide size by scaleFactor.
            if scale_factor > 1.0 {
                size /= scale_factor;
            }
        }

        // Step 9: Return an effective visual size result with size set to size,
        // width set to width, and height set to height.
        Some(size)
    }

    /// Collects an image LCP candidate during paint traversal.
    /// Evaluation is deferred until `compute_new_lcp_candidate`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn collect_image_record(
        &mut self,
        tag: Tag,
        bounds: LayoutRect,
        clip_rect: LayoutRect,
        transform: FastLayoutTransform,
        url: Option<ServoUrl>,
        natural_width: Option<Au>,
        natural_height: Option<Au>,
    ) {
        self.painted_images.push(ImageRecord {
            tag,
            bounds,
            clip_rect,
            transform,
            url,
            natural_width,
            natural_height,
        });
    }

    /// Accumulates a text fragment's rect into the per-element record.
    /// The fragment rect is provided in containing-block space; the
    /// cumulative transform maps it to world space here.
    /// Called per fragment during paint traversal.
    /// Evaluation is deferred until `compute_new_lcp_candidate`.
    pub(crate) fn accumulate_text_rect(
        &mut self,
        tag: Tag,
        rect: LayoutRect,
        transform: FastLayoutTransform,
    ) {
        let border_box = transform_f32_rectangle(rect.to_rect(), transform)
            .unwrap_or_default()
            .to_box2d();
        self.painted_text_nodes
            .entry(tag.node)
            .and_modify(|record| record.rects.push(border_box))
            .or_insert(TextRecord {
                tag,
                rects: vec![border_box],
            });
    }

    /// <https://www.w3.org/TR/largest-contentful-paint/#compute-a-new-largest-contentful-paint-candidate>
    ///
    /// Evaluates all collected image and text records and selects the
    /// largest contentful paint candidate. Called once after paint
    /// traversal, before the display list is sent.
    ///
    /// Implements spec §4.2: iterates `paintedImages` then
    /// `paintedTextNodes`, gating each element as "reported exactly once".
    pub(crate) fn compute_new_lcp_candidate(&mut self) {
        let mut largest_size = self.lcp_size;
        let mut best: Option<(Tag, Option<ServoUrl>, f32)> = None;

        // Spec §4.2: "For each record of paintedImages".
        for record in std::mem::take(&mut self.painted_images) {
            if !self.reported_image_nodes.insert(record.tag.node) {
                continue;
            }
            // Step 4.3: intersectionRect.
            let intersection_rect =
                transform_f32_rectangle(record.clip_rect.to_rect(), record.transform)
                    .unwrap_or_default()
                    .intersection(&self.viewport_rect.to_rect())
                    .map(|rect| rect.to_box2d())
                    .unwrap_or_default();
            // Step 4.4: effective visual size.
            let Some(size) = self.effective_visual_size(
                record.bounds,
                record.clip_rect,
                intersection_rect,
                record.transform,
                record.natural_width,
                record.natural_height,
            ) else {
                continue;
            };
            if size > largest_size {
                largest_size = size;
                best = Some((record.tag, record.url, size));
            }
        }

        // Spec §4.2: "For each textNode of paintedTextNodes".
        for (node, record) in std::mem::take(&mut self.painted_text_nodes) {
            if !self.reported_text_nodes.insert(node) {
                continue;
            }
            // Spec: "union of the border boxes of all Text nodes".
            let union_rect = record
                .rects
                .into_iter()
                .reduce(|a, b| a.union(&b))
                .unwrap_or_default();
            // Step 4.3: intersectionRect (identity — rects are world-space).
            let intersection_rect = union_rect
                .intersection(&self.viewport_rect)
                .unwrap_or_default();
            // Step 4.4: effective visual size.
            let Some(size) = self.effective_visual_size(
                union_rect,
                union_rect,
                intersection_rect,
                FastLayoutTransform::identity(),
                None,
                None,
            ) else {
                continue;
            };
            if size > largest_size {
                largest_size = size;
                best = Some((record.tag, None, size));
            }
        }

        let Some((tag, url, size)) = best else { return };
        self.lcp_size = largest_size;

        let uuid = self.lcp_next_uuid;
        self.lcp_next_uuid += 1;
        self.current_lcp_node = Some(tag.node);
        self.lcp_candidate = Some(LCPCandidate::new(LCPCandidateID(uuid), size as usize, url));
        self.lcp_candidate_updated = true;
    }

    pub(crate) fn did_lcp_candidate_update(&self) -> bool {
        self.lcp_candidate_updated
    }

    pub(crate) fn unset_lcp_candidate_updated(&mut self) {
        self.lcp_candidate_updated = false;
    }

    pub(crate) fn largest_contentful_paint_candidate(&self) -> Option<LCPCandidate> {
        self.lcp_candidate.clone()
    }

    pub(crate) fn current_lcp_node(&self) -> Option<OpaqueNode> {
        self.current_lcp_node
    }
}
