/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use paint_api::largest_contentful_paint_candidate::{LCPCandidate, LargestContentfulPaint};
use rustc_hash::{FxHashMap, FxHashSet};
use servo_base::cross_process_instant::CrossProcessInstant;
use servo_base::id::WebViewId;
use webrender_api::PipelineId;

/// Holds the [`LargestContentfulPaintsContainer`] for each pipeline.
#[derive(Default)]
pub(crate) struct LargestContentfulPaintCalculator {
    lcp_containers: FxHashMap<PipelineId, LargestContentfulPaintsContainer>,
    disabled_webviews: FxHashSet<WebViewId>,
}

impl LargestContentfulPaintCalculator {
    pub(crate) fn new() -> Self {
        Self {
            lcp_containers: Default::default(),
            disabled_webviews: Default::default(),
        }
    }

    pub(crate) fn append_lcp_candidate(
        &mut self,
        candidate: LCPCandidate,
        pipeline_id: PipelineId,
        webview_id: &WebViewId,
    ) {
        assert!(self.enabled_for_webview(webview_id));
        // Layout already selects the winner — we just keep the latest.
        self.lcp_containers
            .entry(pipeline_id)
            .or_default()
            .pending_lcp = Some(candidate);
    }

    pub(crate) fn enabled_for_webview(&self, webview_id: &WebViewId) -> bool {
        !self.disabled_webviews.contains(webview_id)
    }

    pub(crate) fn remove_lcp_candidates_for_pipeline(&mut self, pipeline_id: &PipelineId) {
        self.lcp_containers.remove(pipeline_id);
    }

    pub(crate) fn calculate_largest_contentful_paint(
        &mut self,
        paint_time: CrossProcessInstant,
        pipeline_id: PipelineId,
    ) -> Option<LargestContentfulPaint> {
        self.lcp_containers
            .get_mut(&pipeline_id)
            .and_then(|container| container.calculate_largest_contentful_paint(paint_time))
    }

    /// <https://www.w3.org/TR/largest-contentful-paint/#limitations>
    pub(crate) fn disable_for_webview(&mut self, webview_id: WebViewId) {
        self.disabled_webviews.insert(webview_id);
    }

    pub(crate) fn enable_for_webview(&mut self, webview_id: &WebViewId) {
        self.disabled_webviews.remove(webview_id);
    }
}

/// Holds the pending LCP candidate and the latest LCP for a pipeline.
#[derive(Default)]
struct LargestContentfulPaintsContainer {
    /// The most recent LCP candidate received from layout. Layout already
    /// selects the candidate with the largest effective visual size, so we
    /// only need to compare against the persisted winner.
    pending_lcp: Option<LCPCandidate>,
    /// The most recent Largest Contentful Paint, if any.
    latest_lcp: Option<LargestContentfulPaint>,
}

impl LargestContentfulPaintsContainer {
    fn calculate_largest_contentful_paint(
        &mut self,
        paint_time: CrossProcessInstant,
    ) -> Option<LargestContentfulPaint> {
        let Some(candidate) = self.pending_lcp.take() else {
            return self.latest_lcp.clone();
        };
        if self
            .latest_lcp
            .as_ref()
            .map_or(true, |l| candidate.area > l.area)
        {
            self.latest_lcp = Some(LargestContentfulPaint::from(candidate, paint_time));
        }
        self.latest_lcp.clone()
    }
}
