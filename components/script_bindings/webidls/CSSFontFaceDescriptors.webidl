/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * The origin of this IDL file is
 * https://www.w3.org/TR/css-fonts-4/#om-fontface
 */

[Exposed=Window]
interface CSSFontFaceDescriptors : CSSStyleDeclaration {
  attribute [LegacyNullToEmptyString] CSSOMString src;
  attribute [LegacyNullToEmptyString] CSSOMString fontFamily;
  attribute [LegacyNullToEmptyString] CSSOMString font-family;
  attribute [LegacyNullToEmptyString] CSSOMString fontStyle;
  attribute [LegacyNullToEmptyString] CSSOMString font-style;
  attribute [LegacyNullToEmptyString] CSSOMString fontWeight;
  attribute [LegacyNullToEmptyString] CSSOMString font-weight;
  attribute [LegacyNullToEmptyString] CSSOMString fontStretch;
  attribute [LegacyNullToEmptyString] CSSOMString font-stretch;
  attribute [LegacyNullToEmptyString] CSSOMString fontWidth;
  attribute [LegacyNullToEmptyString] CSSOMString font-width;
  attribute [LegacyNullToEmptyString] CSSOMString unicodeRange;
  attribute [LegacyNullToEmptyString] CSSOMString unicode-range;
};