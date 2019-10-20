/**
 * Java Settlers - An online multiplayer version of the game Settlers of Catan
 * This file Copyright (C) 2013-2014,2018 Jeremy D Monin <jeremy@nand.net>
 * Portions of this file Copyright (C) 2017 Ruud Poutsma <rtimon@gmail.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * The maintainer of this program can be reached at jsettlers@nand.net
 **/
package soc.game;

import soc.util.SOCStringManager;

/**
 * A single Dev Card, probably within a player's {@link SOCInventory}.
 *
 * @author Jeremy D Monin &lt;jeremy@nand.net&gt;
 * @since 2.0.00
 */
public class SOCDevCard
    extends SOCInventoryItem implements SOCDevCardConstants  // SOCInventoryItem implies Cloneable
{
    /**
     * the set of resources a player needs to buy a development card.
     *<P>
     * Before v2.0.00 this field was {@code SOCGame.CARD_SET}.
     *
     * @see SOCPlayingPiece#getResourcesToBuild(int)
     * @see SOCInventory
     */
    public static final SOCResourceSet COST = new SOCResourceSet(0, 1, 1, 1, 0, 0);

    /**
     * If true, {@link #getItemName(SOCGame, boolean, SOCStringManager)} can just use the
     * {@link SOCInventoryItem#strKey strKey} or {@link SOCInventoryItem#aStrKey aStrKey}
     * keys already looked up for the constructor super call.
     *<P>
     * If false, the item text varies by game options and must be calculated each time
     * {@link #getItemName(SOCGame, boolean, SOCStringManager)} is called.
     */
    final private boolean nameKeyPrecalc;

    /**
     * Is this card type a Victory Point card?
     * @param devCardType  A constant such as {@link SOCDevCardConstants#UNIV}
     *               or {@link SOCDevCardConstants#ROADS}
     * @return  True for VP card types, false otherwise
     * @see #isVPItem()
     */
    public static boolean isVPCard(final int devCardType)
    {
        return (devCardType >= SOCDevCardConstants.CAP) && (devCardType <= SOCDevCardConstants.CHAPEL);
    }

    /**
     * Resource type-and-count text keys for {@link #getCardTypeName(int, SOCGame, boolean, SOCStringManager)}.
     * Each subarray's indexes are the same values as constants in range {@link SOCDevCardConstants#UNKNOWN}
     * to {@link SOCDevCardConstants#KNIGHT}.
     */
    private static final String[][] GETCARDTYPENAME_KEYS =
    {
        {     // without article
            "spec.dcards.unknown", "spec.dcards.roadbuilding", "spec.dcards.discoveryplenty", "spec.dcards.monopoly",
            "spec.dcards.capgovhouse", "spec.dcards.market", "spec.dcards.university",
            "spec.dcards.temple", "spec.dcards.towerchapel", "spec.dcards.knightsoldier"
        }, {  // with article (a/an)
            "spec.dcards.aunknown", "spec.dcards.aroadbuilding", "spec.dcards.adiscoveryplenty", "spec.dcards.amonopoly",
            "spec.dcards.acapgovhouse", "spec.dcards.amarket", "spec.dcards.auniversity",
            "spec.dcards.atemple", "spec.dcards.atowerchapel", "spec.dcards.aknightsoldier"
        }
    };

    /**
     * Get a card type's name key.
     * @param devCardType  A constant such as {@link SOCDevCardConstants#UNIV}
     *               or {@link SOCDevCardConstants#ROADS}
     * @param game  Game data, or {@code null}; some game options might change a card name.
     *              For example, {@link SOCGameOption#K_SC_PIRI _SC_PIRI} renames "Knight" to "Warship".
     * @param withArticle  If true, format is: "a Market (+1VP)"; if false, is "Market (1VP)"
     * @return  The card name key for {@code ctype} and {@code withArticle};
     *     unknown ctypes return "spec.dcards.unknown" / "spec.dcards.aunknown".
     * @see #getCardTypeName(int, SOCGame, boolean, SOCStringManager)
     */
    public static String getCardTypeNameKey
        (final int devCardType, final SOCGame game, final boolean withArticle)
    {
        // i18n: These names are also currently hardcoded in SOCServer.DEBUG_COMMANDS_HELP and .DEBUG_COMMANDS_HELP_DEV_TYPES

        final String ctname;

        if ((devCardType == SOCDevCardConstants.KNIGHT) && (game != null) && game.isGameOptionSet(SOCGameOption.K_SC_PIRI))
        {
            return (withArticle) ? "spec.dcards.aknightsoldier.warship" : "spec.dcards.knightsoldier.warship";
        }

        final String[] keyArr = GETCARDTYPENAME_KEYS[(withArticle) ? 1 : 0];
        if ((devCardType >= 0) && (devCardType < keyArr.length))
        {
            ctname = keyArr[devCardType];
        } else {
            ctname = keyArr[SOCDevCardConstants.UNKNOWN];
        }

        return ctname;
    }

    /**
     * Get a card type's name.
     * @param devCardType  A constant such as {@link SOCDevCardConstants#UNIV}
     *               or {@link SOCDevCardConstants#ROADS}
     * @param game  Game data, or {@code null}; some game options might change a card name.
     *              For example, {@link SOCGameOption#K_SC_PIRI _SC_PIRI} renames "Knight" to "Warship".
     * @param withArticle  If true, format is: "a Market (+1VP)"; if false, is "Market (1VP)"
     * @param strings  StringManager to get i18n localized text
     * @return  The localized card name, formatted per {@code ctype} and {@code withArticle};
     *          unknown ctypes return "Unknown card type #"
     * @see #getCardTypeNameKey(int, SOCGame, boolean)
     */
    public static String getCardTypeName
        (final int devCardType, final SOCGame game, final boolean withArticle, final SOCStringManager strings)
    {
        final String ctname;

        if ((devCardType >= 0) && (devCardType < GETCARDTYPENAME_KEYS[0].length))
        {
            ctname = strings.get(getCardTypeNameKey(devCardType, game, withArticle));
        } else {
            ctname = "Unknown card type " + devCardType;  // don't bother I18N, should not occur
        }

        return ctname;
    }

    /**
     * Create a new card.
     * @param type   Card type, such as {@link SOCDevCardConstants#ROADS}
     * @param isNew  Is this card newly given to a player, or old from a previous turn?
     */
    public SOCDevCard(final int type, final boolean isNew)
    {
        this(type, isVPCard(type), isNew);
    }

    /** constructor to call super, to avoid 3 isVPCard(type) calls */
    private SOCDevCard(final int type, final boolean isVPCard, final boolean isNew)
    {
        super(type, ! (isNew || isVPCard), isVPCard, isVPCard, false,
              getCardTypeNameKey(type, null, false), getCardTypeNameKey(type, null, true));
        nameKeyPrecalc =
            (type > SOCDevCardConstants.UNKNOWN) && (type < GETCARDTYPENAME_KEYS[0].length)
            && (type != SOCDevCardConstants.KNIGHT);  // KNIGHT changes with game option _SC_PIRI
    }

    /**
     * Get a human-readable description, including type, isPlayable, isVP.
     * @return String of form: "SOCDevCard{type=__, playable=__, isVP=__}"
     * @since 2.0.00
     */
    public String toString()
    {
        return "SOCDevCard{type=" + itype + ", playable=" + isPlayable() + ", isVP=" + isVPItem() + "}";
    }

    //
    // Methods from SOCInventoryItem:
    //  (see there for javadoc)
    //

    public String getItemName
        (final SOCGame game, final boolean withArticle, final SOCStringManager strings)
    {
        if (nameKeyPrecalc)
            return strings.get((withArticle) ? aStrKey : strKey);
        else
            return getCardTypeName(itype, game, withArticle, strings);
    }

}
