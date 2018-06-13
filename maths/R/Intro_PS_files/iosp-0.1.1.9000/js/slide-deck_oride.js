/*
  Still using slide-deck.js from the IOSlides library
  Here we will override the function preparing the elements to be built
  The new function ignores <style> tags as well as .show class elements
*/

SlideDeck.prototype.makeBuildLists_ = function () {
  for (var i = this.curSlide_, slide; slide = this.slides[i]; ++i) {
    var items = slide.querySelectorAll('.build > *:not(style):not(.show)');
    for (var j = 0, item; item = items[j]; ++j) {
      if (item.classList) {
        item.classList.add('to-build');
        if (item.parentNode.classList.contains('fade')) {
          item.classList.add('fade');
        }
      }
    }
  }
};
