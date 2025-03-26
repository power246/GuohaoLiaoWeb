package student;

/**
 * This class is for creating the AI player.
 * It is a "random" AI and the parent of BiggestCardAI
 * and SmallestCarAI.
 *
 * @author Paul Liao
 */
public class AI {
    /**
     * Pick the first valid card to play.
     *
     * @param hand This is the hand cards for this player.
     * @param cardPile This is the cards in the card pile.
     * @return Card This returns the card been chosen.
     */
    public Card getPlay(Hand hand, CardPile cardPile) {
        for (int i = 0; i < hand.getSize(); i++) {
            if (cardPile.canPlay(hand.get(i))) {
                return hand.get(i);
            }
        }
        return null;
    }

    /**
     * The name of AI: Random Card AI.
     *
     * @return String This returns The name of AI.
     */
    @Override
    public String toString() {
        return "Random Card AI";
    }
}
