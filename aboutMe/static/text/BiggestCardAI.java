package student;

/**
 * This class is for creating the AI player.
 * It is the Biggest Card AI and the subclass of AI.
 *
 * @author Paul Liao
 */
public class BiggestCardAI extends AI {

    /**
     * Pick the biggest valid card to play.
     *
     * @param hand This is the hand cards for this player.
     * @param cardPile This is the cards in the card pile.
     * @return Card This returns the card been chosen.
     */
    @Override
    public Card getPlay(Hand hand, CardPile cardPile) {
        boolean flag = true;
        Card card = new Card(1, 1);
        for (int i = 0; i < hand.getSize(); i++) {
            if (cardPile.canPlay(hand.get(i))) {
                flag = false;
                if (card.getRankNum() <= hand.get(i).getRankNum()) {
                    card = hand.get(i);
                }
            }
        }
        if (flag) {
            return null;
        }
        return card;
    }

    /**
     * The name of AI: Biggest Card AI.
     *
     * @return String This returns The name of AI.
     */
    @Override
    public String toString() {
        // The name of AI
        return "Biggest Card AI";
    }
}
