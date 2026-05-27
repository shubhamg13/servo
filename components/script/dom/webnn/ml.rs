/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

//! <https://webmachinelearning.github.io/webnn/#api-ml>

use std::rc::Rc;

use dom_struct::dom_struct;
use script_bindings::reflector::{Reflector, reflect_dom_object};

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{MLContextOptions, MLMethods};
use crate::dom::bindings::reflector::DomGlobal;
use crate::dom::bindings::root::DomRoot;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mlcontext::MLContext;
use crate::realms::InRealm;
use crate::script_runtime::CanGc;

#[dom_struct]
pub(crate) struct ML {
    reflector_: Reflector,
}

impl ML {
    pub(crate) fn new_inherited() -> ML {
        ML {
            reflector_: Reflector::new(),
        }
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<ML> {
        reflect_dom_object(Box::new(ML::new_inherited()), global, can_gc)
    }
}

impl MLMethods<crate::DomTypeHolder> for ML {
    /// <https://webmachinelearning.github.io/webnn/#dom-ml-createcontext>
    fn CreateContext(
        &self,
        _options: &MLContextOptions,
        comp: InRealm,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        let global = &self.global();
        let context = MLContext::new(global, can_gc);
        let promise = Promise::new_in_current_realm(comp, can_gc);
        promise.resolve_native(&context, can_gc);
        promise
    }
}
