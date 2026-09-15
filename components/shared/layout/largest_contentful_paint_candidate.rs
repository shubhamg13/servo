/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! Definitions for the largest-contentful-paint candidate.

use malloc_size_of_derive::MallocSizeOf;
use serde::{Deserialize, Serialize};
use servo_base::cross_process_instant::CrossProcessInstant;
use servo_base::id::LCPCandidateID;
use servo_url::ServoUrl;
use style::dom::OpaqueNode;

/// A largest-contentful-paint candidate
///
/// <https://www.w3.org/TR/largest-contentful-paint/#largest-contentful-paint-candidate>
#[derive(Clone, Debug, Deserialize, MallocSizeOf, Serialize)]
pub struct LCPCandidate {
    /// A unique identifier for this candidate.
    pub id: LCPCandidateID,
    /// <https://www.w3.org/TR/largest-contentful-paint/#largest-contentful-paint-candidate-size>
    pub size: usize,
    /// <https://www.w3.org/TR/largest-contentful-paint/#largest-contentful-paint-candidate-width>
    pub width: usize,
    /// <https://www.w3.org/TR/largest-contentful-paint/#largest-contentful-paint-candidate-height>
    pub height: usize,
    /// <https://www.w3.org/TR/largest-contentful-paint/#largestcontentfulpaint-url>
    pub url: Option<ServoUrl>,
    /// The time at which image finished loading, None for text candidates.
    pub load_time: Option<CrossProcessInstant>,
    /// For <https://www.w3.org/TR/largest-contentful-paint/#largest-contentful-paint-candidate-element>
    /// The DOM node of the candidate's element, if any.
    pub node: Option<OpaqueNode>,
}

impl LCPCandidate {
    pub fn new(
        id: LCPCandidateID,
        size: usize,
        width: usize,
        height: usize,
        url: Option<ServoUrl>,
        node: Option<OpaqueNode>,
        load_time: Option<CrossProcessInstant>,
    ) -> Self {
        Self {
            id,
            size,
            width,
            height,
            url,
            node,
            load_time,
        }
    }
}
